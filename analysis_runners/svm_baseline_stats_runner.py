"""
This script performs the statistical analysis for the SVM parameter tuning (Step 4).
It assumes that 'svm_baseline_runner.py' has already been run and
saved its results to a .json file.

This script reads that .json file and:
1.  Runs a Friedman test to check for *any* significant difference.
2.  If significant, runs a Nemenyi post-hoc test to find the best
    configuration and all configurations *not* significantly
    worse than the best.
3.  Saves a bar chart, CD diagram, and summary tables.
4.  Selects a final "winner" (best accuracy, then lowest time).
"""

import numpy as np
import pandas as pd
import json
from scipy import stats
import scikit_posthocs as sp
from typing import Dict, Any, List
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import os

# --- Configuration ---
try:
    # This works when run as a script
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for interactive/notebook use
    SCRIPT_DIR = Path.cwd()

"""
This script performs the statistical analysis for the SVM parameter tuning (Step 4).
It assumes that 'svm_baseline_runner.py' has already been run and
saved its results to a .json file.

This script reads that .json file and:
1.  Runs a Friedman test to check for *any* significant difference.
2.  If significant, runs a Nemenyi post-hoc test to find the best
    configuration and all configurations *not* significantly
    worse than the best.
3.  Saves a bar chart, CD diagram, and summary tables.
4.  Selects a final "winner" (best accuracy, then lowest time).
"""

import numpy as np
import pandas as pd
import json
from scipy import stats
import scikit_posthocs as sp
from typing import Dict, Any, List
from pathlib import Path
import sys
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
RESULTS_DIR = PROJECT_ROOT / "results" / "svm_baseline"
# This is where the plots and final CSVs will be saved
OUTPUT_DIR = RESULTS_DIR / "statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['adult', 'pen-based']
ALPHA = 0.05  # Significance level
TOP_N_PLOT = 10  # Plot the top 10 configs


# ---------------------

# --- Plotting Functions ---

def plot_top_ranks_bar_chart(avg_ranks: pd.Series, dataset_name: str, top_n: int):
    """
    Plots a horizontal bar chart of the top N algorithms by average rank.
    """
    print(f"  Generating Top {top_n} Bar Chart for {dataset_name}...")
    try:
        sorted_ranks = avg_ranks.sort_values(ascending=True)
        top_n_ranks = sorted_ranks.head(top_n)

        plt.figure(figsize=(10, top_n * 0.4))
        plt.barh(top_n_ranks.index, top_n_ranks.values, color='cornflowerblue')
        plt.gca().invert_yaxis()  # Best at top
        plt.xlabel("Average Rank (Lower is Better)")
        plt.title(f"Top {top_n} SVM Configs by Avg. Rank ({dataset_name})")
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

        fig = plt.figure(figsize=(14, max(7, top_n * 0.3)))
        ax = fig.add_subplot(111)

        sp.critical_difference_diagram(
            ranks=filtered_ranks,
            sig_matrix=filtered_sig_matrix,
            ax=ax,
            label_props={'fontsize': 9}
        )

        ax.set_title(f"Top {top_n} SVM Configs CD Diagram ({dataset_name})", pad=20)
        plt.tight_layout()

        filename = f"{dataset_name}_cd_diagram_top_{top_n}.png"
        full_path = OUTPUT_DIR / filename
        plt.savefig(full_path, bbox_inches='tight')
        print(f"  Saved Top {top_n} CD diagram to {full_path.relative_to(PROJECT_ROOT)}")
        plt.close()

    except Exception as e:
        print(f"  Error generating truncated CD diagram: {e}")


# --- End Plotting Functions ---


def load_tuning_results(dataset_name: str) -> Dict[str, Any]:
    """
    Loads the JSON file containing the raw fold-by-fold results
    from the SVM parameter tuning.
    """
    # Find the JSON file in the results directory
    json_file = RESULTS_DIR / f"svm_params_{dataset_name}_results.json"

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"[Stats-SVM] Loaded '{json_file.name}': {len(data['config_names'])} configs, "
              f"{len(data['accuracy_matrix'])} folds.")
        return data
    except FileNotFoundError:
        print(f"Error: Results file not found: {json_file}")
        print("Please run '2_svm_baseline_runner.py' first to generate this file.")
        raise
    except Exception as e:
        print(f"Error loading {json_file.name}: {e}")
        raise


def run_friedman_test(accuracy_matrix: np.ndarray, alpha: float = 0.05) -> tuple[bool, float, float]:
    """
    Performs the Friedman test.
    """
    config_arrays = [accuracy_matrix[:, i] for i in range(accuracy_matrix.shape[1])]
    statistic, p_value = stats.friedmanchisquare(*config_arrays)
    proceed_to_posthoc = p_value < alpha
    return proceed_to_posthoc, statistic, p_value


def run_nemenyi_test(
        accuracy_matrix: np.ndarray,
        avg_ranks: pd.Series,
        mean_accuracies: pd.Series,
        config_names: List[str],
        summary_data: Dict[str, Any],
        alpha: float = 0.05
) -> tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Performs the Nemenyi post-hoc test.
    """
    df = pd.DataFrame(accuracy_matrix, columns=config_names)
    nemenyi_matrix = sp.posthoc_nemenyi_friedman(df.to_numpy())
    nemenyi_matrix.columns = df.columns
    nemenyi_matrix.index = df.columns

    best_config_name = mean_accuracies.idxmax()
    p_values_vs_best = nemenyi_matrix[best_config_name]
    finalist_names = p_values_vs_best[p_values_vs_best > alpha].index.tolist()

    finalist_data = []
    for config in finalist_names:
        finalist_data.append({
            'Configuration': config,
            'Mean_Accuracy': mean_accuracies[config],
            'Avg_Rank': avg_ranks[config],
            'Mean_Time': summary_data[config]['mean_time'],
            'P_vs_Best': p_values_vs_best[config]
        })

    df_finalists = pd.DataFrame(finalist_data)
    df_finalists = df_finalists.sort_values(
        by=['Mean_Accuracy', 'Mean_Time'],
        ascending=[False, True]
    )
    winner_row = df_finalists.iloc[0]
    return winner_row.to_dict(), df_finalists, nemenyi_matrix


def analyze_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Orchestrates the full statistical analysis for one dataset.
    """
    print(f"\n{'-' * 60}\n[Analysis] Starting analysis for: {dataset_name.upper()}\n{'-' * 60}")

    # 1. Load results from JSON
    data = load_tuning_results(dataset_name)
    config_names = data['config_names']
    accuracy_matrix = np.array(data['accuracy_matrix'])
    summary_data = data['summary']

    # 2. Run Friedman test
    proceed, friedman_stat, friedman_p = run_friedman_test(accuracy_matrix)
    print(f"  Friedman Test: chi2={friedman_stat:.4f}, p-value={friedman_p:.6e}")

    friedman_df = pd.DataFrame([{'metric': 'Accuracy', 'chi2': friedman_stat, 'p-value': friedman_p}])
    friedman_df.to_csv(OUTPUT_DIR / f"{dataset_name}_friedman_test.csv", index=False)

    # 3. Calculate Average Ranks (for plotting)
    # Higher accuracy = better rank (ascending=False)
    ranks_df = pd.DataFrame(accuracy_matrix, columns=config_names)
    avg_ranks = ranks_df.rank(axis=1, ascending=False, method='average').mean()
    avg_ranks = avg_ranks.sort_values()  # Sort by rank (lower is better)
    avg_ranks.to_csv(OUTPUT_DIR / f"{dataset_name}_avg_ranks.csv")
    print(f"  Saved average ranks to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")

    # 4. If significant, run post-hoc
    if proceed:
        print(f"  Result: Significant difference found (p < {ALPHA}). Proceeding to post-hoc.")
        mean_accuracies = ranks_df.mean()

        winner, finalists_df, nemenyi_matrix = run_nemenyi_test(
            accuracy_matrix, avg_ranks, mean_accuracies, config_names, summary_data
        )

        # Save finalist table
        finalists_df.to_csv(OUTPUT_DIR / f"{dataset_name}_finalist_table.csv", index=False)
        print("  Finalist table (statistically equivalent to best):")
        print(finalists_df.to_string(index=False, float_format="%.4f"))

        # 5. Generate Plots
        plot_top_ranks_bar_chart(avg_ranks, dataset_name, top_n=TOP_N_PLOT)
        plot_cd_diagram(avg_ranks, nemenyi_matrix, dataset_name, top_n=TOP_N_PLOT)

        return {
            'dataset': dataset_name,
            'friedman_significant': True,
            'winner': winner,
            'finalists': finalists_df.to_dict('records')
        }
    else:
        # If not significant, just pick the best by mean
        print(f"  Result: No significant difference found (p >= {ALPHA}).")
        mean_accuracies = pd.Series(
            [summary_data[c]['mean_accuracy'] for c in config_names],
            index=config_names
        )
        best_config = mean_accuracies.idxmax()

        print(f"  Selecting best by mean: {best_config} (Acc={mean_accuracies[best_config]:.4f})")

        # Still generate bar plot of ranks
        plot_top_ranks_bar_chart(avg_ranks, dataset_name, top_n=TOP_N_PLOT)

        return {
            'dataset': dataset_name,
            'friedman_significant': False,
            'winner': {
                'Configuration': best_config,
                'Mean_Accuracy': mean_accuracies[best_config],
                'Mean_Time': summary_data[best_config]['mean_time']
            }
        }


if __name__ == "__main__":
    print("\n--- SVM Parameter Tuning: Statistical Analysis (Step 4) ---")

    all_winners: Dict[str, Any] = {}

    for dataset in DATASETS:
        try:
            result = analyze_dataset(dataset)
            all_winners[dataset] = result
        except Exception as e:
            print(f"\n!!! ERROR processing {dataset}: {e} !!!")
            import traceback

            traceback.print_exc()

    # --- Final Summary ---
    print(f"\n{'-' * 60}\n[Analysis] FINAL SUMMARY: Best SVM Configurations\n{'-' * 60}")

    if not all_winners:
        print("No results to summarize.")
    else:
        for ds, res in all_winners.items():
            if 'winner' in res:
                w = res['winner']
                print(f"  Dataset: {ds.upper()}")
                print(f"    Config:  {w['Configuration']}")
                print(f"    Acc:     {w['Mean_Accuracy']:.4f}")
                print(f"    Time:    {w.get('Mean_Time', float('nan')):.3f}s\n")

    print("[Stats] Analysis complete.")

DATASETS = ['adult', 'pen-based']
ALPHA = 0.05  # Significance level
TOP_N_PLOT = 10  # Plot the top 10 configs


# ---------------------

# --- Plotting Functions ---

def plot_top_ranks_bar_chart(avg_ranks: pd.Series, dataset_name: str, top_n: int):
    """
    Plots a horizontal bar chart of the top N algorithms by average rank.
    """
    print(f"  Generating Top {top_n} Bar Chart for {dataset_name}...")
    try:
        sorted_ranks = avg_ranks.sort_values(ascending=True)
        top_n_ranks = sorted_ranks.head(top_n)

        plt.figure(figsize=(10, top_n * 0.4))
        plt.barh(top_n_ranks.index, top_n_ranks.values, color='cornflowerblue')
        plt.gca().invert_yaxis()  # Best at top
        plt.xlabel("Average Rank (Lower is Better)")
        plt.title(f"Top {top_n} SVM Configs by Avg. Rank ({dataset_name})")
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

        fig = plt.figure(figsize=(14, max(7, top_n * 0.3)))
        ax = fig.add_subplot(111)

        sp.critical_difference_diagram(
            ranks=filtered_ranks,
            sig_matrix=filtered_sig_matrix,
            ax=ax,
            label_props={'fontsize': 9}
        )

        ax.set_title(f"Top {top_n} SVM Configs CD Diagram ({dataset_name})", pad=20)
        plt.tight_layout()

        filename = f"{dataset_name}_cd_diagram_top_{top_n}.png"
        full_path = OUTPUT_DIR / filename
        plt.savefig(full_path, bbox_inches='tight')
        print(f"  Saved Top {top_n} CD diagram to {full_path.relative_to(PROJECT_ROOT)}")
        plt.close()

    except Exception as e:
        print(f"  Error generating truncated CD diagram: {e}")


# --- End Plotting Functions ---


def load_tuning_results(dataset_name: str) -> Dict[str, Any]:
    """
    Loads the JSON file containing the raw fold-by-fold results
    from the SVM parameter tuning.
    """
    # Find the JSON file in the results directory
    json_file = RESULTS_DIR / f"svm_params_{dataset_name}_results.json"

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"[Stats-SVM] Loaded '{json_file.name}': {len(data['config_names'])} configs, "
              f"{len(data['accuracy_matrix'])} folds.")
        return data
    except FileNotFoundError:
        print(f"Error: Results file not found: {json_file}")
        print("Please run '2_svm_baseline_runner.py' first to generate this file.")
        raise
    except Exception as e:
        print(f"Error loading {json_file.name}: {e}")
        raise


def run_friedman_test(accuracy_matrix: np.ndarray, alpha: float = 0.05) -> tuple[bool, float, float]:
    """
    Performs the Friedman test.
    """
    config_arrays = [accuracy_matrix[:, i] for i in range(accuracy_matrix.shape[1])]
    statistic, p_value = stats.friedmanchisquare(*config_arrays)
    proceed_to_posthoc = p_value < alpha
    return proceed_to_posthoc, statistic, p_value


def run_nemenyi_test(
        accuracy_matrix: np.ndarray,
        avg_ranks: pd.Series,
        mean_accuracies: pd.Series,
        config_names: List[str],
        summary_data: Dict[str, Any],
        alpha: float = 0.05
) -> tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Performs the Nemenyi post-hoc test.
    """
    df = pd.DataFrame(accuracy_matrix, columns=config_names)
    nemenyi_matrix = sp.posthoc_nemenyi_friedman(df.to_numpy())
    nemenyi_matrix.columns = df.columns
    nemenyi_matrix.index = df.columns

    best_config_name = mean_accuracies.idxmax()
    p_values_vs_best = nemenyi_matrix[best_config_name]
    finalist_names = p_values_vs_best[p_values_vs_best > alpha].index.tolist()

    finalist_data = []
    for config in finalist_names:
        finalist_data.append({
            'Configuration': config,
            'Mean_Accuracy': mean_accuracies[config],
            'Avg_Rank': avg_ranks[config],
            'Mean_Time': summary_data[config]['mean_time'],
            'P_vs_Best': p_values_vs_best[config]
        })

    df_finalists = pd.DataFrame(finalist_data)
    df_finalists = df_finalists.sort_values(
        by=['Mean_Accuracy', 'Mean_Time'],
        ascending=[False, True]
    )
    winner_row = df_finalists.iloc[0]
    return winner_row.to_dict(), df_finalists, nemenyi_matrix


def analyze_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Orchestrates the full statistical analysis for one dataset.
    """
    print(f"\n{'-' * 60}\n[Analysis] Starting analysis for: {dataset_name.upper()}\n{'-' * 60}")

    # 1. Load results from JSON
    data = load_tuning_results(dataset_name)
    config_names = data['config_names']
    accuracy_matrix = np.array(data['accuracy_matrix'])
    summary_data = data['summary']

    # 2. Run Friedman test
    proceed, friedman_stat, friedman_p = run_friedman_test(accuracy_matrix)
    print(f"  Friedman Test: chi2={friedman_stat:.4f}, p-value={friedman_p:.6e}")

    friedman_df = pd.DataFrame([{'metric': 'Accuracy', 'chi2': friedman_stat, 'p-value': friedman_p}])
    friedman_df.to_csv(OUTPUT_DIR / f"{dataset_name}_friedman_test.csv", index=False)

    # 3. Calculate Average Ranks (for plotting)
    # Higher accuracy = better rank (ascending=False)
    ranks_df = pd.DataFrame(accuracy_matrix, columns=config_names)
    avg_ranks = ranks_df.rank(axis=1, ascending=False, method='average').mean()
    avg_ranks = avg_ranks.sort_values()  # Sort by rank (lower is better)
    avg_ranks.to_csv(OUTPUT_DIR / f"{dataset_name}_avg_ranks.csv")
    print(f"  Saved average ranks to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")

    # 4. If significant, run post-hoc
    if proceed:
        print(f"  Result: Significant difference found (p < {ALPHA}). Proceeding to post-hoc.")
        mean_accuracies = ranks_df.mean()

        winner, finalists_df, nemenyi_matrix = run_nemenyi_test(
            accuracy_matrix, avg_ranks, mean_accuracies, config_names, summary_data
        )

        # Save finalist table
        finalists_df.to_csv(OUTPUT_DIR / f"{dataset_name}_finalist_table.csv", index=False)
        print("  Finalist table (statistically equivalent to best):")
        print(finalists_df.to_string(index=False, float_format="%.4f"))

        # 5. Generate Plots
        plot_top_ranks_bar_chart(avg_ranks, dataset_name, top_n=TOP_N_PLOT)
        plot_cd_diagram(avg_ranks, nemenyi_matrix, dataset_name, top_n=TOP_N_PLOT)

        return {
            'dataset': dataset_name,
            'friedman_significant': True,
            'winner': winner,
            'finalists': finalists_df.to_dict('records')
        }
    else:
        # If not significant, just pick the best by mean
        print(f"  Result: No significant difference found (p >= {ALPHA}).")
        mean_accuracies = pd.Series(
            [summary_data[c]['mean_accuracy'] for c in config_names],
            index=config_names
        )
        best_config = mean_accuracies.idxmax()

        print(f"  Selecting best by mean: {best_config} (Acc={mean_accuracies[best_config]:.4f})")

        # Still generate bar plot of ranks
        plot_top_ranks_bar_chart(avg_ranks, dataset_name, top_n=TOP_N_PLOT)

        return {
            'dataset': dataset_name,
            'friedman_significant': False,
            'winner': {
                'Configuration': best_config,
                'Mean_Accuracy': mean_accuracies[best_config],
                'Mean_Time': summary_data[best_config]['mean_time']
            }
        }


if __name__ == "__main__":
    print("\n--- SVM Parameter Tuning: Statistical Analysis (Step 4) ---")

    all_winners: Dict[str, Any] = {}

    for dataset in DATASETS:
        try:
            result = analyze_dataset(dataset)
            all_winners[dataset] = result
        except Exception as e:
            print(f"\n!!! ERROR processing {dataset}: {e} !!!")
            import traceback

            traceback.print_exc()

    # --- Final Summary ---
    print(f"\n{'-' * 60}\n[Analysis] FINAL SUMMARY: Best SVM Configurations\n{'-' * 60}")

    if not all_winners:
        print("No results to summarize.")
    else:
        for ds, res in all_winners.items():
            if 'winner' in res:
                w = res['winner']
                print(f"  Dataset: {ds.upper()}")
                print(f"    Config:  {w['Configuration']}")
                print(f"    Acc:     {w['Mean_Accuracy']:.4f}")
                print(f"    Time:    {w.get('Mean_Time', float('nan')):.3f}s\n")

    print("[Stats] Analysis complete.")