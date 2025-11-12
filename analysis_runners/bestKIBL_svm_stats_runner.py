"""
This script performs the final statistical comparison between the
"Best K-IBL Algorithm" and the "Best SVM Algorithm" (Step 4).

It does the following:
1.  Loads the DETAILED, fold-level results from:
    - 'get_best_kibl_runner.py'
    - 'svm_baseline_runner.py' (which saves a JSON file)
2.  Uses a predefined dictionary (BEST_CONFIGS) to find the 10
    fold-level accuracy scores for the best K-IBL and best SVM.
3.  Performs a paired t-test (stats.ttest_rel) to see if there is a
    statistically significant difference between the two algorithms.
4.  Saves a summary table of the comparison.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from pathlib import Path
import sys
from typing import Dict, Any, List

# --- Configuration ---
try:
    # This works when run as a script
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for interactive/notebook use
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_KIBL_DIR = PROJECT_ROOT / "results" / "kibl_baseline"
RESULTS_SVM_DIR = PROJECT_ROOT / "results" / "svm_baseline"
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find the latest K-IBL detailed results file
try:
    RESULTS_KIBL_FILE = max(RESULTS_KIBL_DIR.glob("kibl_detailed_fold_results_FINAL_*.csv"))
    print(f"[Stats-Compare] Using K-IBL results: {RESULTS_KIBL_FILE.name}")
except ValueError:
    print(f"Error: No K-IBL results file found in {RESULTS_KIBL_DIR}")
    print("Please run '1_get_best_kibl_runner.py' first.")
    sys.exit(1)

# --- Best Algorithm Definitions ---
# Define the "best" configurations found in your previous analyses.
# !! UPDATE THE 'svm' section after running svm_baseline_stats_runner.py !!
BEST_CONFIGS: Dict[str, Dict[str, Any]] = {
    'adult': {
        'kibl': {
            "K": 7, "Distance": "HEOM",
            "Voting": "ModPlurality", "Retention": "NR"
        },
        'svm': {
            "k": "rbf",
            "C": 10,
            "g": 1.0
        }
    },
    'pen-based': {
        'kibl': {
            "K": 5, "Distance": "Euclidean",
            "Voting": "BordaCount", "Retention": "NR"
        },
        'svm': {
            "k": "rbf",
            "C": 10,
            "g": "scale"
        }
    }
}

DATASETS = ['adult', 'pen-based']
ALPHA = 0.05  # Significance level
# ---------------------

def load_kibl_scores(filepath: Path, dataset: str, config: Dict[str, Any]) -> pd.Series:
    """Loads the 10 fold scores for the best k-IBL config."""
    df = pd.read_csv(filepath)

    # Build a filter mask based on the config
    mask = (df['Dataset'].str.lower() == dataset.lower())
    for key, value in config.items():
        mask &= (df[key].apply(lambda x: str(x) == str(value)))

    scores = df[mask].sort_values(by='Fold')['Accuracy']

    if len(scores) != 10:
        print(f"Error: Found {len(scores)} scores for k-IBL config, expected 10.")
        print(f"Config: {config}")
        return pd.Series(dtype=float)

    return scores.reset_index(drop=True)

def load_svm_scores(results_dir: Path, dataset: str, config: Dict[str, Any]) -> pd.Series:
    """Loads the 10 fold scores for the best SVM config from the JSON file."""
    json_file = results_dir / f"svm_params_{dataset}_results.json"

    if not json_file.exists():
        print(f"Error: SVM results JSON not found at {json_file}")
        print("Please run '2_svm_baseline_runner.py' first.")
        return pd.Series(dtype=float)

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Find the column index for this config
    config_name = f"k={config['k']},C={config['C']},g={config['g']}"

    try:
        col_idx = data['config_names'].index(config_name)
    except ValueError:
        print(f"Error: SVM config '{config_name}' not found in JSON file.")
        print("Please run '2_svm_baseline_stats_runner.py' to find the winner, then update BEST_CONFIGS.")
        return pd.Series(dtype=float)

    # Get the accuracy scores for that column (all 10 folds)
    accuracy_matrix = np.array(data['accuracy_matrix'])
    scores = accuracy_matrix[:, col_idx]

    if len(scores) != 10:
        print(f"Error: Found {len(scores)} scores for SVM config, expected 10.")
        return pd.Series(dtype=float)

    return pd.Series(scores)

def main():
    """
    Main function to run the K-IBL vs. SVM comparison.
    """
    print("\n--- Statistical Comparison: Best K-IBL vs. Best SVM (Step 4) ---")

    all_results = []

    for dataset in DATASETS:
        print(f"\n[Stats-Compare] Processing dataset: {dataset.upper()}")

        try:
            # 1. Get the 10 fold scores for each algorithm
            kibl_config = BEST_CONFIGS[dataset]['kibl']
            svm_config = BEST_CONFIGS[dataset]['svm']

            kibl_scores = load_kibl_scores(RESULTS_KIBL_FILE, dataset, kibl_config)
            svm_scores = load_svm_scores(RESULTS_SVM_DIR, dataset, svm_config)

            if kibl_scores.empty or svm_scores.empty:
                print("  Skipping dataset due to data loading error.")
                continue

            # 2. Calculate means
            mean_kibl = kibl_scores.mean()
            mean_svm = svm_scores.mean()

            # 3. Run the paired t-test
            t_stat, p_val = stats.ttest_rel(kibl_scores, svm_scores)

            # 4. Determine the winner
            if np.isnan(p_val):
                decision = "Test Failed"
            elif p_val < ALPHA:
                if t_stat > 0: # K-IBL > SVM
                    decision = "K-IBL is significantly better"
                else: # SVM > K-IBL
                    decision = "SVM is significantly better"
            else:
                decision = "No significant difference"

            print(f"  K-IBL Mean Acc: {mean_kibl:.4f}")
            print(f"  SVM Mean Acc:   {mean_svm:.4f}")
            print(f"  t-statistic={t_stat:.4f}, p-value={p_val:.6f}")
            print(f"  Decision (α=0.05): {decision}")

            all_results.append({
                'Dataset': dataset,
                'KIBL_Config': str(kibl_config),
                'SVM_Config': f"k={svm_config['k']},C={svm_config['C']},g={svm_config['g']}",
                'KIBL_Mean_Acc': mean_kibl,
                'SVM_Mean_Acc': mean_svm,
                't_statistic': t_stat,
                'p_value': p_val,
                'Decision': decision
            })

        except Exception as e:
            print(f"  An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()

    # --- Save final comparison ---
    summary_df = pd.DataFrame(all_results)
    output_file = OUTPUT_DIR / "kibl_vs_svm_comparison.csv"
    summary_df.to_csv(output_file, index=False, float_format="%.6f")

    print("\n" + "=" * 80)
    print(f"Final K-IBL vs. SVM comparison saved to: {output_file.relative_to(PROJECT_ROOT)}")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("\n[Stats-Compare] Done.")


if __name__ == "__main__":
    main()