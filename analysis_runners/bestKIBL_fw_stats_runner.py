"""
This script performs the statistical comparison between the
"Best Baseline k-IBL" and the "Feature Weighting (FW) k-IBL" (Step 5).

It does the following:
1.  Loads the DETAILED fold-level results from:
    - '1_get_best_kibl_runner.py' (for the baseline scores)
    - '3_feature_weighting_runner.py' (for the FW scores)
2.  For each dataset and each FW method (mutual_info, relieff):
    a. Extracts the 10 baseline scores.
    b. Extracts the 10 corresponding FW scores.
    c. Performs a paired Wilcoxon signed-rank test.
    d. Calculates effect size |r|.
3.  Applies a Holm-Bonferroni p-value correction (since we make 2 comparisons).
4.  Saves the final comparison table to a CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from scipy import stats
import sys

# --- Path Configuration ---
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_KIBL_DIR = PROJECT_ROOT / "results" / "kibl_baseline"
RESULTS_FW_DIR = PROJECT_ROOT / "results" / "feature_weighting_complete" # Use 'complete' for final analysis
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find the latest results files
try:
    KIBL_FOLDS_FILE = max(RESULTS_KIBL_DIR.glob("kibl_detailed_fold_results_FINAL_*.csv"))
    FW_FOLDS_FILE = max(RESULTS_FW_DIR.glob("fw_detailed_fold_results_*.csv"))
    print(f"[Stats-FW] Using Baseline K-IBL file: {KIBL_FOLDS_FILE.name}")
    print(f"[Stats-FW] Using FW K-IBL file: {FW_FOLDS_FILE.name}")
except ValueError as e:
    print(f"Error: Missing results file. {e}")
    print("Please run '1_get_best_kibl_runner.py' and '3_feature_weighting_runner.py' first.")
    sys.exit(1)

# --- PARAMETERS ---
DATASETS = ["adult", "pen-based"]
FW_METHODS = ["mutual_info", "relieff"]
ALPHA = 0.05  # Significance level

# Define the *exact* baseline configs we want to compare against
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
# --- END CONFIG ---


# ----------------------------
# Helper Functions
# ----------------------------

def holm_adjust(pvals: List[float]) -> List[float]:
    """
    Performs the Holm-Bonferroni step-down procedure
    to correct p-values for multiple comparisons.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals) # Number of comparisons
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    prev = 0.0

    for i, idx in enumerate(order):
        adj = (m - i) * pvals[idx]
        adj = max(prev, adj) # Enforce monotonicity
        adj = min(1.0, adj)  # Cap at 1.0
        adjusted[idx] = adj
        prev = adj
    return adjusted.tolist()


def load_data(baseline_path: Path, fw_path: Path) -> Dict[str, pd.DataFrame]:
    """Load baseline and FW data from their respective paths."""
    print("[Stats-FW] Loading data...")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    base_df = pd.read_csv(baseline_path)

    if not fw_path.exists():
        raise FileNotFoundError(f"FW file not found: {fw_path}")
    fw_df = pd.read_csv(fw_path)
    return {"base": base_df, "fw": fw_df}


def get_scores_by_config(df: pd.DataFrame, dataset: str, config: Dict[str, Any]) -> pd.Series:
    """
    Filters a detailed results file to get the 10-fold
    accuracy scores for a specific configuration.
    """
    mask = (df['Dataset'].str.lower() == dataset.lower())
    for key, value in config.items():
        # Special check for Weight_Method which is in fw_df but not baseline_conf
        if key in df.columns:
            mask &= (df[key].apply(lambda x: str(x) == str(value)))

    scores = df[mask].sort_values(by='Fold')['Accuracy']

    if len(scores) != 10:
        print(f"Warning: Found {len(scores)} scores, expected 10.")
        print(f"Config: {config}")
        return pd.Series(dtype=float)

    return scores.reset_index(drop=True)


# --- Main execution ---
def main():
    """
    Main script logic: Loops through all datasets.
    """
    print("\n--- Statistical Comparison: Best k-IBL vs. FW-k-IBL (Step 5) ---")

    all_comparison_results = []

    try:
        # 1. Load data
        dfs = load_data(KIBL_FOLDS_FILE, FW_FOLDS_FILE)
    except FileNotFoundError:
        sys.exit(1)

    for dataset in DATASETS:
        print(f"\n[Stats-FW] Processing: {dataset.upper()}")

        try:
            # 2. Get the 10 baseline scores
            baseline_conf = BASELINE_CONFIGS[dataset]
            baseline_scores = get_scores_by_config(dfs["base"], dataset, baseline_conf)

            if baseline_scores.empty:
                print(f"  Error: Could not find baseline config. Skipping {dataset}.")
                print(f"  Searched for: {baseline_conf}")
                continue

            print(f"  Baseline Mean Acc: {baseline_scores.mean():.4f}")

            comparison_rows = []

            # 3. Compare against each FW method
            for method in FW_METHODS:
                print(f"  Comparing vs. FW Method: {method}")

                # The FW runner uses the same baseline config, just adds a weight method
                # We filter the fw_df using the baseline config *and* the weight method
                fw_config_filter = {**baseline_conf, "Weight_Method": method}
                fw_scores = get_scores_by_config(dfs["fw"], dataset, fw_config_filter)

                if fw_scores.empty:
                    print(f"    Error: Could not find FW config for {method}. Skipping.")
                    print(f"    Searched for: {fw_config_filter}")
                    continue

                # 4. Run Wilcoxon signed-rank test (paired test)
                # We use Wilcoxon as it's more robust to non-normal distributions than t-test
                diffs = fw_scores - baseline_scores
                w_stat, p = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", mode="auto")

                # 5. Calculate effect size |r|
                n = len(diffs)
                if np.isnan(p):
                    z, r = np.nan, np.nan
                else:
                    # Calculate z-score from p-value for effect size
                    z = stats.norm.ppf(1 - p/2.0) * np.sign(diffs.mean())
                    r = abs(z) / np.sqrt(n) # r = |Z| / sqrt(N)

                comparison_rows.append({
                    "method": method,
                    "mean_fw": fw_scores.mean(),
                    "mean_base": baseline_scores.mean(),
                    "mean_diff": diffs.mean(),
                    "p_value": p,
                    "effect_size_r": r,
                })

            if not comparison_rows:
                print(f"  No FW methods to compare for {dataset}.")
                continue

            # 6. Apply Holm-Bonferroni correction
            res = pd.DataFrame(comparison_rows)
            pvals = res["p_value"].fillna(1.0).tolist()
            res["p_holm"] = holm_adjust(pvals)

            # --- Format for Final Report ---
            res["Dataset"] = dataset
            res["mean_fw(%)"] = (res["mean_fw"] * 100).round(4)
            res["mean_base(%)"] = (res["mean_base"] * 100).round(4)
            res["Δ points (%)"] = (res["mean_diff"] * 100).round(4)
            res["p (Wilcoxon)"] = res["p_value"].round(6)
            res["p (Holm)"] = res["p_holm"].round(6)
            res["|r| effect"] = res["effect_size_r"].round(3)

            res["decision_vs_baseline"] = np.where(
                (res["p_holm"] < ALPHA) & (res["mean_diff"] > 0), "Significantly better",
                np.where(
                    (res["p_holm"] < ALPHA) & (res["mean_diff"] < 0), "Significantly worse",
                    "No significant difference"
                ),
            )

            order_cols = [
                "Dataset", "method", "mean_fw(%)", "mean_base(%)", "Δ points (%)",
                "p (Wilcoxon)", "p (Holm)", "|r| effect", "decision_vs_baseline",
            ]
            all_comparison_results.append(res[order_cols].sort_values("p (Holm)"))

        except Exception as e:
            print(f"\n--- ERROR processing {dataset} ---")
            print(f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # --- Save final summary ---
    if not all_comparison_results:
        print("\n[Stats-FW] No results to save.")
        return

    final_df = pd.concat(all_comparison_results, ignore_index=True)
    output_file = OUTPUT_DIR / "kibl_vs_fw_comparison.csv"
    final_df.to_csv(output_file, index=False, float_format="%.6f")

    print("\n" + "=" * 80)
    print(f"Final k-IBL vs. FW comparison saved to: {output_file.relative_to(PROJECT_ROOT)}")
    print("=" * 80)
    print(final_df.to_string(index=False))
    print("\n[Stats-FW] All FW statistical analyses complete.")

if __name__ == "__main__":
    main()