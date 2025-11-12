"""
This script performs the statistical analysis to find the
"best k-IBL algorithm" for each dataset (Step 3).

It does the following:
1.  Loads the detailed fold-level results from the
    'get_best_kibl_runner.py' script.
2.  For each dataset, it creates a (fold x configuration) matrix.
3.  It performs a Friedman test to check for significant
    differences among all k-IBL configurations.
4.  If significant, it runs a Nemenyi post-hoc test to find
    the "best" configuration and its statistical equals.
5.  It saves a bar chart of the top ranks and a CD diagram.
"""

import pandas as pd
import numpy as np
import scikit_posthocs as sp
from scipy import stats
from pathlib import Path
import sys
import glob
import matplotlib.pyplot as plt
import os

# --- Configuration ---
try:
    # This works when run as a script
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for interactive/notebook use
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "kibl_baseline"
# This is where the plots and final CSVs will be saved
OUTPUT_DIR = RESULTS_DIR / "statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find the latest detailed results file
try:
    RESULTS_FILE_PATH = max(RESULTS_DIR.glob("kibl_detailed_fold_results_FINAL_*.csv"))
    print(f"[Stats-KIBL] Using results file: {RESULTS_FILE_PATH.name}")
except ValueError:
    print(f"Error: No results file found in {RESULTS_DIR}")
    print("Please run 'get_best_kibl_runner.py' first.")
    sys.exit(1)

DATASETS = ['adult', 'pen-based']
ALPHA = 0.05  # Significance level
TOP_N_PLOT = 15 # Plot the top 15 configs in the bar chart
# ---------------------

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads the detailed fold-level results CSV."""
    if not filepath.exists():
        print(f"Error: Results file not found at {filepath}")
        sys.exit(1)

    print(f"[Stats-KIBL] Loading results from {filepath.name}...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} rows.")
    return df


def create_fold_matrix(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Pivots the dataframe to create a matrix where:
    - Rows = Folds (0-9)
    - Columns = Configurations (e.g., "K=3,Euclidean,...")
    - Values = Accuracy
    """
    print(f"\n[Stats-KIBL] Processing dataset: {dataset.upper()}")

    df_ds = df[df['Dataset'].str.lower() == dataset.lower()].copy()

    if df_ds.empty:
        print(f"  Error: No data found for dataset '{dataset}'.")
        return pd.DataFrame()

    # Create a unique name for each configuration
    df_ds['Config_Name'] = (
            "K=" + df_ds['K'].astype(str) + ", " +
            df_ds['Distance'] + ", " +
            df_ds['Voting'] + ", " +
            df_ds['Retention']
    )

    try:
        fold_matrix = df_ds.pivot(
            index='Fold',
            columns='Config_Name',
            values='Accuracy'
        )
    except Exception as e:
        print(f"  Error pivoting data: {e}")
        return pd.DataFrame()

    # Check for missing values (e.g., a config failed on one fold)
    if fold_matrix.isnull().values.any():
        print("  Warning: Missing values detected. Imputing with 0.0 for stability.")
        fold_matrix = fold_matrix.fillna(0.0)

    return fold_matrix


def plot_top_ranks_bar_chart(avg_ranks: pd.Series, dataset_name: str, top_n: int):
    """
    Plots a horizontal bar chart of the top N algorithms by average rank.
    """
    print(f"  Generating Top {top_n} Bar Chart for {dataset_name}...")
    try:
        # Sort so the best (lowest) ranks are first
        sorted_ranks = avg_ranks.sort_values(ascending=True)
        top_n_ranks = sorted_ranks.head(top_n)

        plt.figure(figsize=(10, top_n * 0.4))
        plt.barh(top_n_ranks.index, top_n_ranks.values, color='c')
        plt.gca().invert_yaxis() # Best at top
        plt.xlabel("Average Rank (Lower is Better)")
        plt.title(f"Top {top_n} k-IBL Configs by Avg. Rank ({dataset_name})")
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        for index, value in enumerate(top_n_ranks):
            plt.text(value, index, f' {value:.2f}', va='center')

        filename = f"{dataset_name}_bar_chart_top_{top_n}_ranks.png"
        full_path = OUTPUT_DIR / filename
        plt.savefig(full_path, bbox_inches='tight')
        print(f"  Saved bar chart to {full_path.relative_to(PROJECT_ROOT)}")
        plt.close()

    except Exception as e:
        print(f"  Error generating bar chart: {e}")


def plot_cd_diagram(avg_ranks: pd.Series, nemenyi_results_df: pd.DataFrame, dataset_name: str, top_n: int):
    """
    Plots a truncated Critical Difference diagram for the Top N algorithms.
    """
    print(f"  Generating Top {top_n} Critical Difference Diagram for {dataset_name}...")

    try:
        sorted_ranks = avg_ranks.sort_values(ascending=True)
        filtered_ranks = sorted_ranks.head(top_n)
        top_n_names = filtered_ranks.index

        # Filter the full Nemenyi p-value matrix to only these top N
        filtered_sig_matrix = nemenyi_results_df.loc[top_n_names, top_n_names]

        print(f"  Plotting CD diagram for the top {len(filtered_ranks)} configurations.")

        fig = plt.figure(figsize=(14, max(7, top_n * 0.3))) # Dynamic height
        ax = fig.add_subplot(111)

        sp.critical_difference_diagram(
            ranks=filtered_ranks,
            sig_matrix=filtered_sig_matrix,
            ax=ax,
            label_props={'fontsize': 9}
        )

        ax.set_title(f"Top {top_n} k-IBL Configs CD Diagram ({dataset_name})", pad=20)
        plt.tight_layout()

        filename = f"{dataset_name}_cd_diagram_top_{top_n}.png"
        full_path = OUTPUT_DIR / filename
        plt.savefig(full_path, bbox_inches='tight')
        print(f"  Saved Top {top_n} CD diagram to {full_path.relative_to(PROJECT_ROOT)}")
        plt.close()

    except Exception as e:
        print(f"  Error generating truncated CD diagram: {e}")
        import traceback
        traceback.print_exc()


def run_statistical_analysis(fold_matrix: pd.DataFrame, dataset: str):
    """
    Runs the Friedman and Nemenyi tests and saves plots/tables.
    """
    num_configs = len(fold_matrix.columns)
    num_folds = len(fold_matrix)
    print(f"  Running Friedman test on {num_configs} configurations across {num_folds} folds...")

    # --- 1. Friedman Test ---
    stat, p_friedman = stats.friedmanchisquare(*[fold_matrix[col] for col in fold_matrix.columns])
    print(f"  Friedman Test: chi2={stat:.4f}, p-value={p_friedman:.6e}")

    # Save Friedman result
    friedman_df = pd.DataFrame([{'metric': 'Accuracy', 'chi2': stat, 'p-value': p_friedman}])
    friedman_df.to_csv(OUTPUT_DIR / f"{dataset}_friedman_test.csv", index=False)

    if p_friedman >= ALPHA:
        print(f"  Result: No significant difference found (p >= {ALPHA}).")
        best_config_name = fold_matrix.mean().idxmax()
        print(f"  Best config (by mean accuracy): {best_config_name}")
        return best_config_name

    print(f"  Result: Significant difference found (p < {ALPHA}). Proceeding to post-hoc...")

    # --- 2. Nemenyi Post-Hoc Test ---
    nemenyi_results = sp.posthoc_nemenyi_friedman(fold_matrix)

    # --- 3. Calculate Average Ranks ---
    # We rank ascending=False (higher accuracy = better rank)
    avg_ranks = fold_matrix.rank(axis=1, ascending=False, method='average').mean()
    avg_ranks.index = fold_matrix.columns
    avg_ranks = avg_ranks.sort_values() # Sort by rank (lower is better)
    avg_ranks.to_csv(OUTPUT_DIR / f"{dataset}_avg_ranks.csv")
    print(f"  Saved average ranks to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")

    # --- 4. Find the Best Configuration ---
    mean_accuracies = fold_matrix.mean().sort_values(ascending=False)
    best_by_acc_name = mean_accuracies.idxmax()

    print(f"\n  Best configuration (by mean accuracy): {best_by_acc_name}")
    print(f"    Mean Accuracy: {mean_accuracies[best_by_acc_name]:.4f}")

    p_values_vs_best = nemenyi_results[best_by_acc_name]
    finalists = p_values_vs_best[p_values_vs_best > ALPHA].index.tolist()

    print(f"  Found {len(finalists)} configurations statistically tied with the best:")

    # Create and save finalist table
    finalist_data = []
    for f in finalists:
        finalist_data.append({
            'Config_Name': f,
            'Mean_Accuracy': mean_accuracies[f],
            'Avg_Rank': avg_ranks[f],
            'p_vs_Best': p_values_vs_best[f]
        })
    finalist_df = pd.DataFrame(finalist_data).sort_values(by='Avg_Rank')
    finalist_df.to_csv(OUTPUT_DIR / f"{dataset}_finalist_table.csv", index=False)

    print(finalist_df.to_string(index=False, float_format="%.4f"))

    # --- 5. Generate Plots ---
    plot_top_ranks_bar_chart(avg_ranks, dataset, top_n=TOP_N_PLOT)
    plot_cd_diagram(avg_ranks, nemenyi_results, dataset, top_n=TOP_N_PLOT)

    return best_by_acc_name


# --- Main execution ---
def main():
    """
    Main function to run the statistical analysis for k-IBL configs.
    """
    print("\n--- K-IBL Configuration Statistical Analysis (Step 3) ---")

    try:
        df_full = load_data(RESULTS_FILE_PATH)
    except (FileNotFoundError, SystemExit):
        return

    best_configs = {}

    for dataset in DATASETS:
        if dataset.lower() not in df_full['Dataset'].str.lower().unique():
            print(f"\nWarning: Dataset '{dataset}' not found in results file. Skipping.")
            continue

        fold_matrix = create_fold_matrix(df_full, dataset)
        if fold_matrix.empty:
            continue

        best_config = run_statistical_analysis(fold_matrix, dataset)
        best_configs[dataset] = best_config

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE: BEST K-IBL CONFIGURATIONS")
    print("=" * 80)
    for dataset, config in best_configs.items():
        print(f"  {dataset.upper():<12}: {config}")
    print("\n[Stats-KIBL] Done.")


if __name__ == "__main__":
    main()