"""
This is the FINAL statistical analysis script (Step 6.d) [cite: 154-156].
It compares the THREE CHAMPION algorithm families against each other:
1.  Best Baseline k-IBL (from 1_get_bestKIBL_stats_runner.py)
2.  Best Feature Weighting (FW) k-IBL (from 4_bestKIBL_fw_stats_runner.py)
3.  Best Instance Reduction (IR) k-IBL (from 5_bestKIBL_ir_stats_runner.py)

It does the following:
1.  Loads the DETAILED fold-level CSV results from all previous experiments.
2.  Builds a (fold x 3 champions) accuracy matrix for each dataset.
3.  Runs a Friedman test to check for overall significance [cite: 20-22].
4.  Runs a Nemenyi post-hoc test to compare the three champions.
5.  Generates and saves a Critical Difference (CD) diagram.
"""

import pandas as pd
import numpy as np
import scikit_posthocs as spc
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, Any

# --- Configuration ---
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_KIBL_DIR = PROJECT_ROOT / "results" / "kibl_baseline"
RESULTS_FW_DIR = PROJECT_ROOT / "results" / "feature_weighting_complete"
RESULTS_IR_DIR = PROJECT_ROOT / "results" / "ir_baseline"

# Directory for final plots
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find the latest results files
try:
    RESULTS_KIBL_FILE = max(RESULTS_KIBL_DIR.glob("kibl_detailed_fold_results_FINAL_*.csv"))
    RESULTS_FW_FILE = max(RESULTS_FW_DIR.glob("fw_detailed_fold_results_*.csv"))
    RESULTS_IR_FILE = max(RESULTS_IR_DIR.glob("ir_detailed_fold_results_*.csv"))
    print("[Stats-Final] Using K-IBL results:", RESULTS_KIBL_FILE.name)
    print("[Stats-Final] Using FW results:", RESULTS_FW_FILE.name)
    print("[Stats-Final] Using IR results:", RESULTS_IR_FILE.name)
except ValueError as e:
    print(f"Error: Missing detailed results file. {e}")
    print("Please run all experiment runners (kibl, fw, ir) first.")
    sys.exit(1)
# ---------------------

# --- Algorithm Selection ---
# Define the "best" algorithms from previous steps
# !! UPDATE THESE with the winners from your stats scripts !!
BEST_ALGOS: Dict[str, Dict[str, Any]] = {
    'adult': {
        # Best baseline config for 'adult' (from script #1)
        'Baseline': {"K": 7, "Distance": "HEOM", "Voting": "ModPlurality", "Retention": "NR"},
        # Best FW method for 'adult' (from script #4)
        'FW': 'mutual_info',  # Placeholder: UPDATE THIS after running script #4
        # Best IR method for 'adult' (from script #5)
        'IR': 'ENN',  # Placeholder: UPDATE THIS after running script #5
    },
    'pen-based': {
        # Best baseline config for 'pen-based' (from script #1)
        'Baseline': {"K": 5, "Distance": "Euclidean", "Voting": "BordaCount", "Retention": "NR"},
        # Best FW method for 'pen-based' (from script #4)
        'FW': 'none',  # Placeholder: UPDATE THIS after running script #4
        # Best IR method for 'pen-based' (from script #5)
        'IR': 'ENN',  # Placeholder: UPDATE THIS after running script #5
    }
}

DATASETS = ['adult', 'pen-based']
ALPHA = 0.05


# ---------------------

def load_all_data() -> Dict[str, pd.DataFrame]:
    """Loads all required CSVs."""
    print("[Stats-Final] Loading all result files...")
    try:
        df_kibl = pd.read_csv(RESULTS_KIBL_FILE)
        df_fw = pd.read_csv(RESULTS_FW_FILE)
        df_ir = pd.read_csv(RESULTS_IR_FILE)
        print("[Stats-Final] All files loaded successfully.")
        return {"kibl": df_kibl, "fw": df_fw, "ir": df_ir}
    except FileNotFoundError as e:
        print(f"Error: Could not load file. {e}")
        print("Please update the paths in the Configuration section.")
        sys.exit(1)


def build_comparison_matrix(data: Dict[str, pd.DataFrame], dataset: str) -> pd.DataFrame:
    """
    Builds the final (fold x 3 champions) accuracy matrix for one dataset.
    This is the required input format for the Friedman test.
    """
    print(f"\n[Stats-Final] Building comparison matrix for: {dataset.upper()}")

    matrix = pd.DataFrame(index=range(10))  # 10 folds
    config = BEST_ALGOS[dataset]
    baseline_conf = config['Baseline']

    # --- 1. Get Baseline Champion ---
    try:
        baseline_mask = (data['kibl']['Dataset'].str.lower() == dataset.lower())
        for key, value in baseline_conf.items():
            baseline_mask &= (data['kibl'][key].apply(lambda x: str(x) == str(value)))
        baseline_acc = data['kibl'][baseline_mask].sort_values('Fold')['Accuracy']
        if len(baseline_acc) != 10: raise ValueError("Baseline scores not found")
        matrix['Baseline'] = baseline_acc.reset_index(drop=True)
    except Exception as e:
        print(f"  Error finding Baseline scores: {e}. Config: {baseline_conf}")
        return pd.DataFrame()  # Return empty df

    # --- 2. Get Feature Weighting (FW) Champion (UPDATED) ---
    fw_method = config.get('FW') # Use .get() for safety
    if fw_method is None or str(fw_method).lower() == 'none':
        print("  Skipping FW champion (config is 'none' or None).")
    else:
        try:
            # The FW runner uses the *same baseline config*
            fw_conf = {**baseline_conf, "Weight_Method": fw_method}
            fw_mask = (data['fw']['Dataset'].str.lower() == dataset.lower())
            for key, value in fw_conf.items():
                if key in data['fw'].columns:
                    fw_mask &= (data['fw'][key].apply(lambda x: str(x) == str(value)))

            fw_acc = data['fw'][fw_mask].sort_values('Fold')['Accuracy']
            if len(fw_acc) != 10: raise ValueError(f"FW scores not found for {fw_method}")
            matrix[f"FW_{fw_method}"] = fw_acc.reset_index(drop=True)
        except Exception as e:
            print(f"  Error finding FW scores: {e}. Config: {fw_method}")
            return pd.DataFrame()

    # --- 3. Get Instance Reduction (IR) Champion (UPDATED) ---
    ir_method = config.get('IR') # Use .get() for safety
    if ir_method is None or str(ir_method).lower() == 'none':
        print("  Skipping IR champion (config is 'none' or None).")
    else:
        try:
            # The IR runner also uses the *same baseline config*
            ir_conf = {**baseline_conf, "IR_Technique": ir_method}
            ir_mask = (data['ir']['Dataset'].str.lower() == dataset.lower())
            for key, value in ir_conf.items():
                if key in data['ir'].columns:
                    ir_mask &= (data['ir'][key].apply(lambda x: str(x) == str(value)))

            ir_acc = data['ir'][ir_mask].sort_values('Fold')['Accuracy']
            if len(ir_acc) != 10: raise ValueError(f"IR scores not found for {ir_method}")
            matrix[ir_method] = ir_acc.reset_index(drop=True)
        except Exception as e:
            print(f"  Error finding IR scores: {e}. Config: {ir_method}")
            return pd.DataFrame()

    if matrix.isnull().values.any():
        print("  Warning: Matrix contains missing values! Check configurations.")
        print(matrix)
        matrix = matrix.fillna(0.0)  # Impute NaNs for safety

    if matrix.shape[0] != 10:
        print(f"  Error: Matrix has {matrix.shape[0]} rows, expected 10.")
        return pd.DataFrame()

    print("  Matrix built successfully with columns:", matrix.columns.tolist())
    return matrix


def run_final_stats_and_plot(matrix: pd.DataFrame, dataset: str):
    """
    Runs the final Friedman/Nemenyi tests and generates the CD Diagram.
    """
    print(f"  Running final Friedman test for {dataset}...")

    # --- 1. Friedman Test ---
    stat, p_friedman = stats.friedmanchisquare(*[matrix[col] for col in matrix.columns])
    print(f"  Friedman Test: chi2={stat:.4f}, p-value={p_friedman:.6e}")

    friedman_df = pd.DataFrame([{'metric': 'Accuracy', 'chi2': stat, 'p-value': p_friedman}])
    friedman_df.to_csv(OUTPUT_DIR / f"{dataset}_final_friedman_test.csv", index=False)

    if p_friedman >= ALPHA:
        print("  Result: No significant difference found. Skipping post-hoc.")
        return

    # --- 2. Nemenyi Post-Hoc Test ---
    print("  Result: Significant difference found. Running Nemenyi post-hoc test...")
    nemenyi_results = spc.posthoc_nemenyi_friedman(matrix.to_numpy())
    nemenyi_results.columns = matrix.columns
    nemenyi_results.index = matrix.columns

    nemenyi_path = OUTPUT_DIR / f"final_nemenyi_matrix_{dataset}.csv"
    nemenyi_results.to_csv(nemenyi_path, float_format="%.6f")
    print(f"\n  Nemenyi p-value matrix saved to {nemenyi_path.relative_to(PROJECT_ROOT)}")
    print(nemenyi_results.to_string(float_format="%.4f"))

    # --- 3. Calculate Average Ranks ---
    avg_ranks = matrix.rank(axis=1, ascending=False, method='average').mean()
    avg_ranks = avg_ranks.sort_values()  # Sort by rank (lower is better)
    ranks_path = OUTPUT_DIR / f"final_avg_ranks_{dataset}.csv"
    avg_ranks.to_csv(ranks_path)
    print(f"\n  Average Ranks (lower is better) saved to {ranks_path.relative_to(PROJECT_ROOT)}:")
    print(avg_ranks.to_string(float_format="%.3f"))

    # --- 4. Plot Critical Difference (CD) Diagram ---
    print("  Generating Critical Difference Diagram...")

    fig = plt.figure(figsize=(8, 4))  # Smaller figure for 3 algos
    ax = fig.add_subplot(111)

    spc.critical_difference_diagram(
        ranks=avg_ranks,
        sig_matrix=nemenyi_results,
        ax=ax,
        label_props={'fontsize': 10}
    )

    ax.set_title(f"Final Algorithm Champions CD Diagram ({dataset.upper()}, α=0.05)", pad=20)
    plt.tight_layout()

    # Save the plot
    plot_filename = OUTPUT_DIR / f"final_cd_diagram_{dataset}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"  Saved CD Diagram to: {plot_filename.relative_to(PROJECT_ROOT)}")
    plt.close()


# --- Main execution ---
def main():
    """
    Main function to run the final statistical comparison.
    """
    print("\n--- K-IBL Final Statistical Comparison (Baseline vs. FW vs. IR) ---")

    try:
        data = load_all_data()
    except (FileNotFoundError, SystemExit):
        return  # Error is printed in the load function

    for dataset in DATASETS:
        try:
            # 1. Build the (fold x 3 champions) matrix
            matrix = build_comparison_matrix(data, dataset)
            if matrix.empty or matrix.shape[1] < 2:
                print(f"  Failed to build matrix for {dataset}. Skipping.")
                continue

            # 2. Run stats and plot
            run_final_stats_and_plot(matrix, dataset)

        except KeyError as e:
            print(f"\n  Error: Missing data for {dataset}. {e}")
            print("  Check your 'BEST_ALGOS' config and that all CSVs are up to date.")
        except Exception as e:
            print(f"\n  An error occurred while processing {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print("\n[Stats-Final] Final analysis complete.")


if __name__ == "__main__":
    main()
