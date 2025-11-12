"""
This script performs the statistical analysis to compare the
Best Baseline k-IBL vs. the IR k-IBL techniques (Step 6) [cite: 154-156].

It does the following:
1.  Loads the detailed fold-level results from:
    - '1_get_best_kibl_runner.py' (for the baseline scores)
    - '4_ir_baseline_runner.py' (for ENN, TCNN, ICF scores)
2.  For each dataset, it builds a (fold x algorithm) matrix for:
    - 'Baseline'
    - 'ENN'
    - 'TCNN'
    - 'ICF'
3.  It performs this analysis for Accuracy, Efficiency (Time), and Storage [cite: 133-136].
4.  It performs a Friedman test and Nemenyi post-hoc test for each metric [cite: 20-22].
5.  It saves bar charts, CD diagrams, and summary tables.
"""

import pandas as pd
import scipy.stats as sp_stats
import scikit_posthocs as spc
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from typing import Dict, Any

# --- Configuration ---
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_KIBL_DIR = PROJECT_ROOT / "results" / "kibl_baseline"
RESULTS_IR_DIR = PROJECT_ROOT / "results" / "ir_baseline"
# This is where the plots and final CSVs will be saved
OUTPUT_DIR = PROJECT_ROOT / "results" / "ir_baseline" / "statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find the latest results files
try:
    KIBL_FOLDS_FILE = max(RESULTS_KIBL_DIR.glob("kibl_detailed_fold_results_FINAL_*.csv"))
    IR_FOLDS_FILE = max(RESULTS_IR_DIR.glob("ir_detailed_fold_results_*.csv"))
    print(f"[Stats-IR] Using Baseline K-IBL file: {KIBL_FOLDS_FILE.name}")
    print(f"[Stats-IR] Using IR K-IBL file: {IR_FOLDS_FILE.name}")
except ValueError as e:
    print(f"Error: Missing results file. {e}")
    print("Please run '1_get_best_kibl_runner.py' and '4_ir_baseline_runner.py' first.")
    sys.exit(1)

# Define the *exact* baseline configs
# !! UPDATED with your provided winners !!
BASELINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "adult": {
        "K": 7, "Distance": "HEOM",
        "Voting": "ModPlurality", "Retention": "NR"
    },
    "pen-based": {
        "K": 5, "Distance": "Euclidean",
        "Voting": "BordaCount", "Retention": "NR"
    }
}

DATASETS = ["adult", "pen-based"]
IR_TECHNIQUES = ['ENN', 'TCNN', 'ICF']
METRICS = ['Accuracy', 'Total_Time_s', 'Storage_percent']
ALPHA = 0.05
# ---------------------


def load_data(baseline_path: Path, ir_path: Path) -> Dict[str, pd.DataFrame]:
    """Load baseline and IR data from their respective paths."""
    print("[Stats-IR] Loading data...")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    base_df = pd.read_csv(baseline_path)

    if not ir_path.exists():
        raise FileNotFoundError(f"IR file not found: {ir_path}")
    ir_df = pd.read_csv(ir_path)
    return {"base": base_df, "ir": ir_df}


def build_comparison_matrix(
        dfs: Dict[str, pd.DataFrame],
        dataset: str,
        metric: str
) -> pd.DataFrame:
    """
    Builds the final (fold x algorithm) matrix for one dataset and one metric.
    """
    print(f"  Building matrix for: {dataset.upper()} / {metric}")

    matrix = pd.DataFrame(index=range(10))  # 10 folds
    baseline_conf = BASELINE_CONFIGS[dataset]

    # --- 1. Get Baseline results ---
    baseline_mask = (dfs['base']['Dataset'].str.lower() == dataset.lower())
    for key, value in baseline_conf.items():
        baseline_mask &= (dfs['base'][key].apply(lambda x: str(x) == str(value)))

    base_run_df = dfs['base'][baseline_mask].sort_values(by='Fold')
    if len(base_run_df) != 10:
        print(f"  Warning: Found {len(base_run_df)} scores for Baseline, expected 10.")

    # Map metrics
    if metric == 'Total_Time_s':
         # 'Total_Time_s' for Baseline = 0 (reduction) + prediction time
         # Convert ms/instance to total seconds
         base_time_s = (base_run_df['Time_per_instance_ms'] / 1000.0) * base_run_df['Test_size']
         matrix['Baseline'] = base_time_s.reset_index(drop=True)
    else:
        # Accuracy or Storage_percent
        base_scores = base_run_df[metric]
        matrix['Baseline'] = base_scores.reset_index(drop=True)

    # --- 2. Get Instance Reduction (IR) results ---
    # The IR runner already used the correct baseline config, so we just filter by technique
    for ir_method in IR_TECHNIQUES:
        ir_mask = (
                (dfs['ir']['Dataset'].str.lower() == dataset.lower()) &
                (dfs['ir']['IR_Technique'] == ir_method)
        )
        ir_scores = dfs['ir'][ir_mask].sort_values(by='Fold')[metric]
        if len(ir_scores) != 10:
             print(f"  Warning: Found {len(ir_scores)} scores for {ir_method} {metric}, expected 10.")
        matrix[ir_method] = ir_scores.reset_index(drop=True)

    matrix = matrix.dropna(axis=1, how='any')  # Drop cols that failed to load

    if matrix.isnull().values.any():
        print(f"  Warning: Missing values detected in matrix for {metric}. Imputing 0.0")
        print(matrix)
        matrix = matrix.fillna(0.0)

    return matrix


def plot_top_ranks_bar_chart(avg_ranks: pd.Series, metric_name: str, dataset_name: str):
    """
    Plots a horizontal bar chart of algorithms by average rank.
    """
    print(f"  Generating Bar Chart for {metric_name} ({dataset_name})...")
    try:
        sorted_ranks = avg_ranks.sort_values(ascending=True)

        plt.figure(figsize=(10, len(sorted_ranks) * 0.5))
        plt.barh(sorted_ranks.index, sorted_ranks.values, color='slateblue')
        plt.gca().invert_yaxis()  # Best at top
        plt.xlabel("Average Rank (Lower is Better)")
        plt.title(f"Algorithm Ranks for {metric_name} ({dataset_name})")
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        for index, value in enumerate(sorted_ranks):
            plt.text(value, index, f' {value:.2f}', va='center')

        filename = f"{dataset_name}_bar_chart_ranks_{metric_name}.png"
        full_path = OUTPUT_DIR / filename
        plt.savefig(full_path, bbox_inches='tight')
        print(f"  Saved bar chart to {full_path.relative_to(PROJECT_ROOT)}")
        plt.close()

    except Exception as e:
        print(f"  Error generating bar chart: {e}")


def plot_cd_diagram(avg_ranks: pd.Series, nemenyi_results_df: pd.DataFrame, metric_name: str, dataset_name: str):
    """
    Plots a Critical Difference diagram.
    """
    print(f"  Generating Critical Difference Diagram for {metric_name} ({dataset_name})...")

    try:
        filtered_ranks = avg_ranks.sort_values(ascending=True)
        filtered_sig_matrix = nemenyi_results_df.loc[filtered_ranks.index, filtered_ranks.index]

        fig = plt.figure(figsize=(10, max(4, len(filtered_ranks) * 0.4)))
        ax = fig.add_subplot(111)

        spc.critical_difference_diagram(
            ranks=filtered_ranks,
            sig_matrix=filtered_sig_matrix,
            ax=ax,
            label_props={'fontsize': 10}
        )

        ax.set_title(f"Critical Difference Diagram for {metric_name} ({dataset_name})", pad=20)
        plt.tight_layout()

        filename = f"{dataset_name}_cd_diagram_{metric_name}.png"
        full_path = OUTPUT_DIR / filename
        plt.savefig(full_path, bbox_inches='tight')
        print(f"  Saved CD diagram to {full_path.relative_to(PROJECT_ROOT)}")
        plt.close()

    except Exception as e:
        print(f"  Error generating CD diagram: {e}")


def run_statistical_analysis(matrix: pd.DataFrame, metric: str, dataset: str):
    """
    Runs the Friedman and Nemenyi tests for a single metric.
    """
    num_algos = len(matrix.columns)
    if num_algos < 2:
        print(f"  Only {num_algos} algorithm found. Skipping stats.")
        return None, None

    print(f"  Running Friedman test on {num_algos} algorithms...")

    # --- 1. Friedman Test ---
    stat, p_friedman = sp_stats.friedmanchisquare(*[matrix[col] for col in matrix.columns])
    print(f"  Friedman Test: chi2={stat:.4f}, p-value={p_friedman:.6e}")

    # --- 2. Calculate Ranks ---
    # For Accuracy, higher is better (ascending=False)
    # For Time/Storage, lower values are better (ascending=True)
    lower_is_better = metric in ['Total_Time_s', 'Storage_percent']
    avg_ranks = matrix.rank(axis=1, ascending=not lower_is_better, method='average').mean()
    avg_ranks = avg_ranks.sort_values()  # Sort by rank (lower rank is better)

    # --- 3. Run Post-Hoc and Plots ---
    nemenyi_results = None
    if p_friedman < ALPHA:
        print(f"  Result: Significant difference found (p < {ALPHA}). Running post-hoc...")
        nemenyi_results = spc.posthoc_nemenyi_friedman(matrix.to_numpy())
        nemenyi_results.columns = matrix.columns
        nemenyi_results.index = matrix.columns

        plot_cd_diagram(avg_ranks, nemenyi_results, metric, dataset)
    else:
        print(f"  Result: No significant difference found (p >= {ALPHA}).")

    plot_top_ranks_bar_chart(avg_ranks, metric, dataset)

    return avg_ranks, pd.DataFrame([{'Metric': metric, 'Friedman_chi2': stat, 'p_value': p_friedman}])


# --- Main execution ---
def main():
    pd.set_option('display.max_rows', 100)

    print("\n--- K-IBL vs. IR Statistical Analysis (Step 6) ---")

    try:
        data = load_data(KIBL_FOLDS_FILE, IR_FOLDS_FILE)
    except (FileNotFoundError, SystemExit):
        return

    for dataset in DATASETS:
        print(f"\n{'-'*80}\n[Stats-IR] STARTING ANALYSIS FOR DATASET: {dataset.upper()}\n{'-'*80}")

        all_ranks = {}
        all_friedman = []

        for metric in METRICS:
            print(f"\n--- Analyzing Metric: {metric} ---")

            matrix = build_comparison_matrix(data, dataset, metric)
            if matrix.empty or matrix.shape[1] < 2:
                print(f"  Could not build matrix for {metric}. Skipping.")
                continue

            avg_ranks, friedman_res = run_statistical_analysis(matrix, metric, dataset)

            if avg_ranks is not None:
                all_ranks[metric] = avg_ranks
                all_friedman.append(friedman_res)

        # --- Save Summary Tables ---
        if all_friedman:
            friedman_summary = pd.concat(all_friedman, ignore_index=True)
            f_path = OUTPUT_DIR / f"{dataset}_table_friedman_summary.csv"
            friedman_summary.to_csv(f_path, index=False, float_format="%.6f")
            print(f"\nSaved Friedman summary to {f_path.relative_to(PROJECT_ROOT)}")
            print(friedman_summary.to_string(index=False))

        if all_ranks:
            ranks_summary = pd.DataFrame(all_ranks)
            ranks_summary.columns = [f"{c}_AvgRank" for c in ranks_summary.columns]
            r_path = OUTPUT_DIR / f"{dataset}_table_avg_ranks.csv"
            ranks_summary.to_csv(r_path, float_format="%.3f")
            print(f"\nSaved Avg. Ranks to {r_path.relative_to(PROJECT_ROOT)}")
            print(ranks_summary.sort_values(by='Accuracy_AvgRank').to_string(float_format="%.3f"))

    print("\n[Stats-IR] IR analysis complete.")

if __name__ == "__main__":
    main()