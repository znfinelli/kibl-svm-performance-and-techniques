"""
This script analyzes how the instance reduction of the training set affects the
results obtained by the svm algorithm (Step 6.e).

It does the following:
1.  Loads two JSON result files:
    - The one from 'svm_baseline_runner.py' (Baseline SVM scores).
    - The one from 'svm_ir_runner.py' (IR-SVM scores).
2.  For each dataset ('adult', 'pen-based'):
    a. Extracts the 10 fold scores for the best baseline SVM.
    b. Extracts the 10 fold scores for each IR-SVM (ENN, TCNN, ICF).
3.  For each IR technique, it performs a paired t-test between:
    (10 scores from Baseline-SVM) vs. (10 scores from IR-SVM)
4.  It saves and prints a final summary table of the comparison.
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
    SCRIPT_DIR = Path(__file__).parent.resolve()
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_SVM_BASELINE_DIR = PROJECT_ROOT / "results" / "svm_baseline"
RESULTS_SVM_IR_DIR = PROJECT_ROOT / "results" / "svm_ir_analysis"
OUTPUT_DIR = RESULTS_SVM_IR_DIR / "statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find the latest results files
try:
    RESULTS_SVM_BASELINE_JSON = max(RESULTS_SVM_BASELINE_DIR.glob("svm_params_*_results.json")) # Finds one, assumes we need both
    RESULTS_SVM_IR_JSON = max(RESULTS_SVM_IR_DIR.glob("svm_ir_fold_results_*.json"))
    print(f"[Stats-SVM-IR] Using SVM-IR results: {RESULTS_SVM_IR_JSON.name}")
except ValueError as e:
    print(f"Error: Missing results JSON file. {e}")
    print("Please run 'svm_baseline_runner.py' and 'svm_ir_runner.py' first.")
    sys.exit(1)

# --- Best SVM Definitions ---
# Define the "best" SVM configuration found in your tuning.
# This is used to select the correct "baseline" row.
# !! UPDATE THESE with the winners from 'svm_baseline_stats_runner.py' !!
BEST_SVM_CONFIGS: Dict[str, Dict[str, Any]] = {
    'adult': {
        "k": "rbf",
        "C": 10,
        "g": 1.0
    },
    'pen-based': {
        "k": "rbf",
        "C": 10,
        "g": "scale"
    }
}

IR_TECHNIQUES = ['ENN', 'TCNN', 'ICF']
DATASETS = ['adult', 'pen-based']
ALPHA = 0.05  # Significance level
# ---------------------

def load_json_results(filepath: Path) -> Dict[str, Any]:
    """Loads a single JSON results file."""
    if not filepath.exists():
        print(f"Error: Results file not found at {filepath}")
        raise FileNotFoundError
    print(f"[Stats-SVM-IR] Loading results from {filepath.name}...")
    with open(filepath, 'r') as f:
        return json.load(f)

def get_baseline_svm_scores(dataset: str, config: Dict[str, Any]) -> List[float]:
    """Loads the 10 fold scores for the best baseline SVM config."""
    json_file = RESULTS_SVM_BASELINE_DIR / f"svm_params_{dataset}_results.json"
    data = load_json_results(json_file)

    config_name = f"k={config['k']},C={config['C']},g={config['g']}"

    try:
        col_idx = data['config_names'].index(config_name)
    except ValueError:
        print(f"Error: SVM config '{config_name}' not found in {json_file.name}.")
        return []

    accuracy_matrix = np.array(data['accuracy_matrix'])
    scores = accuracy_matrix[:, col_idx]

    if len(scores) != 10:
        print(f"Error: Found {len(scores)} scores for SVM config, expected 10.")
        return []

    return list(scores)


def main():
    """
    Main function to run the SVM vs. IR-SVM comparison.
    """
    print("\n--- Statistical Comparison: SVM (Baseline) vs. SVM (on IR sets) (Step 6.e) ---")

    try:
        ir_data = load_json_results(RESULTS_SVM_IR_JSON)
    except FileNotFoundError:
        print("Exiting. Please check file paths.")
        return

    all_results = []

    for dataset in DATASETS:
        print(f"\n[Stats-SVM-IR] Processing dataset: {dataset.upper()}")

        # 1. Get the Baseline SVM results (10 scores)
        try:
            svm_config = BEST_SVM_CONFIGS[dataset]
            baseline_scores = get_baseline_svm_scores(dataset, svm_config)
            if not baseline_scores:
                print(f"  Error: Could not find baseline SVM config for {dataset}. Skipping.")
                continue
            baseline_mean = np.mean(baseline_scores)
        except Exception as e:
            print(f"  Error loading baseline scores: {e}. Skipping {dataset}.")
            continue

        print(f"  Baseline SVM Mean Acc: {baseline_mean:.4f}")

        # 2. Iterate through each IR technique
        for ir_tech in IR_TECHNIQUES:
            print(f"  Comparing vs. IR Technique: {ir_tech}")

            # Get the 10 fold scores for the IR-SVM
            try:
                ir_scores = ir_data['datasets'][dataset][ir_tech]['accuracies']
                ir_mean = ir_data['datasets'][dataset][ir_tech]['mean_accuracy']
                if not ir_scores or len(ir_scores) != 10:
                    raise ValueError(f"IR scores missing or incomplete for {ir_tech}")
            except (KeyError, ValueError) as e:
                print(f"    Error: Failed to get IR-SVM scores for {ir_tech}. {e}")
                continue

            # 3. Run the paired t-test
            t_stat, p_val = stats.ttest_rel(baseline_scores, ir_scores)

            # 4. Determine the decision
            if np.isnan(p_val):
                decision = "Test Failed"
            elif p_val < ALPHA:
                if t_stat > 0:  # Baseline > IR
                    decision = "Baseline is significantly better"
                else:  # IR > Baseline
                    decision = "IR-SVM is significantly better"
            else:
                decision = "No significant difference"

            print(f"    IR-SVM Mean Acc:   {ir_mean:.4f}")
            print(f"    t-statistic={t_stat:.4f}, p-value={p_val:.6f}")
            print(f"    Decision (α=0.05): {decision}")

            all_results.append({
                'Dataset': dataset,
                'Comparison': f"Baseline vs. {ir_tech}",
                'Baseline_Mean_Acc': baseline_mean,
                'IR_Mean_Acc': ir_mean,
                't_statistic': t_stat,
                'p_value': p_val,
                'Decision': decision
            })

    # --- Save final comparison ---
    summary_df = pd.DataFrame(all_results)
    output_file = OUTPUT_DIR / "svm_vs_ir_svm_statistical_comparison.csv"
    summary_df.to_csv(output_file, index=False, float_format="%.6f")

    print("\n" + "=" * 80)
    print(f"Final SVM vs. IR-SVM comparison saved to: {output_file.relative_to(PROJECT_ROOT)}")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("\n[Stats-SVM-IR] Done.")


if __name__ == "__main__":
    main()