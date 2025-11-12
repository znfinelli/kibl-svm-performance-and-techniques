"""
This script runs the k-IBL algorithm using feature weights (Step 5).

It can be run in two modes, set by the 'RUN_MODE' variable:
1.  'complete': Uses the *entire* training set to calculate weights.
2.  'subsample': Uses a smaller, random subsample for faster
                 weight calculation (good for testing).

It does the following:
1.  Loads the best k-IBL configuration for each dataset.
2.  For each fold of a dataset:
    a. Calculates feature weights using two methods:
       - 'mutual_info': `sklearn.feature_selection.mutual_info_classif`
       - 'relieff': `skrebate.ReliefF`
    b. Runs the best k-IBL config, but *overrides* the
       default weights with the new calculated weights.
3.  Saves detailed fold-by-fold results and an aggregated summary.
"""

# --- Path Setup ---
from pathlib import Path
import sys
import os

try:
    # This works when run as a script
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for interactive/notebook use
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- End Path Setup ---


import time
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# --- Import Project Modules ---
try:
    from core_modules.parser import load_all_folds
    from core_modules.kibl import kIBLAlgorithm
    from core_modules.distances import euclidean_distance, cosine_distance, heom_distance
    from core_modules.voting import modified_plurality_vote, borda_count_vote
except ImportError as exc:
    print(f"Error: Could not import project modules (parser, kibl, etc.)")
    print(f"Ensure they are in the project root directory: {PROJECT_ROOT}")
    raise

# --- Type Aliases ---
FoldData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
KIBLMetrics = Dict[str, Any]

# --- Configuration ---
RUN_MODE = 'complete'  # <-- 'complete' for final results, 'subsample' for testing
DATA_DIR = PROJECT_ROOT / "data" / "datasetsCBR"

if RUN_MODE == 'subsample':
    RESULTS_DIR = PROJECT_ROOT / "results" / "feature_weighting_subsample"
    SUBSAMPLE_SIZE = 5000
    RELIEFF_NEIGHBORS = 10
else:
    RESULTS_DIR = PROJECT_ROOT / "results" / "feature_weighting_complete"
    SUBSAMPLE_SIZE = None  # Will be ignored
    RELIEFF_NEIGHBORS = 100 # Default for complete run

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# --- End Configuration ---


# --- Experiment Configuration ---
# These are the "best" baseline k-IBL configurations found
# in the first experiment.
# !! UPDATED with your provided winners !!
BEST_CONFIGS: Dict[str, Dict[str, Any]] = {
    'pen-based': {
        'k': 5,
        'distance': 'Euclidean',
        'voting': 'BordaCount',
        'retention': 'NR'
    },
    'adult': {
        'k': 7,
        'distance': 'HEOM',
        'voting': 'ModPlurality',
        'retention': 'NR'
    },
}
# ---------------------------------


def compute_feature_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = 'mutual_info',
    run_mode: str = 'complete',
    subsample_size: int = 5000,
    relieff_neighbors: int = 100,
    random_state: int = 42
) -> np.ndarray:
    """
    Computes feature weights for the given training data based on the run_mode.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    n_samples, n_features = X_train.shape
    rng = np.random.default_rng(random_state)

    # --- Data Preparation (Subsampling) ---
    if run_mode == 'subsample' and subsample_size is not None and n_samples > subsample_size:
        print(f"      (Using subsample n={subsample_size})...", end='', flush=True)
        idx = rng.choice(n_samples, size=subsample_size, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
        sub_n = subsample_size
    else:
        if run_mode == 'subsample':
            print(f"      (Dataset n={n_samples} <= subsample_size, using full)...", end='', flush=True)
        else:
             print(f"      (Using complete n={n_samples})...", end='', flush=True)
        X_sub, y_sub = X_train, y_train
        sub_n = n_samples

    # --- Weight Calculation ---
    start_t = time.time()
    if method == 'mutual_info':
        # mutual_info_classif works well for heterogeneous (mixed) data
        mi = mutual_info_classif(
            X_sub, y_sub,
            discrete_features='auto', # Let sklearn decide
            random_state=random_state
        )
        # Normalize weights to [0, 1] range
        max_val = np.max(mi)
        weights = mi / max_val if max_val > 0 else np.ones(n_features)

    elif method == 'relieff':
        try:
            from skrebate import ReliefF  # type: ignore
        except ImportError as exc:
            print("\n\nError: ReliefF requires the 'scikit-rebate' package.")
            print("Please install it: `pip install scikit-rebate`")
            raise ImportError(
                "ReliefF weighting requires the 'scikit-rebate' package. "
            ) from exc

        # Use the neighbor count appropriate for the mode
        n_neighbors = min(relieff_neighbors, sub_n - 1) if sub_n > 1 else 1
        model = ReliefF(n_neighbors=n_neighbors, n_jobs=-1)

        # Fit the model and get feature importances
        model.fit(X_sub, y_sub)
        importances = model.feature_importances_.astype(float)

        # Normalize importances to [0, 1] range (ReliefF can give negative scores)
        # Shift scores to be non-negative before scaling
        min_val = np.min(importances)
        shifted_importances = importances - min_val
        max_val = np.max(shifted_importances)
        weights = shifted_importances / max_val if max_val > 0 else np.ones(n_features)

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    print(f" done in {time.time() - start_t:.2f}s")
    return weights


class FeatureWeightingExperimentRunner:
    """
    Orchestrates the k-IBL experiments using feature weighting.
    """
    def __init__(self, data_dir: Path, output_dir: Path,
                 weighting_methods: List[str] = None):

        self.data_dir = str(data_dir) # Ensure string paths
        self.output_dir = str(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Set weighting methods to use
        self.weighting_methods = weighting_methods or ['mutual_info', 'relieff']

        # Mapping of distance function names to callables
        self.distance_functions: Dict[str, Callable] = {
            'Euclidean': euclidean_distance,
            'Cosine': cosine_distance,
            'HEOM': heom_distance
        }
        self.voting_functions: Dict[str, Callable] = {
            'ModPlurality': modified_plurality_vote,
            'BordaCount': borda_count_vote
        }

        print(f"[FW Runner] Run Mode: {RUN_MODE.upper()}")
        print(f"[FW Runner] Data directory: {self.data_dir}")
        print(f"[FW Runner] Output directory: {self.output_dir}")
        print(f"[FW Runner] Weighting Methods: {self.weighting_methods}")

    def _evaluate_fold(self,
                       fold_data: FoldData,
                       config: Dict[str, Any],
                       dist_func: Callable,
                       vote_func: Callable,
                       weights: np.ndarray) -> KIBLMetrics:
        """Helper to run a single k-IBL evaluation."""
        X_train, y_train, X_test, y_test = fold_data
        original_train_size = len(X_train)

        # Instantiate k-IBL with the baseline config
        kibl = kIBLAlgorithm(
            k=config['k'],
            distance_func=dist_func,
            voting_func=vote_func,
            retention_policy=config['retention']
        )

        # Fit on the data
        kibl.fit(X_train, y_train)

        # --- KEY STEP: Override the default weights ---
        kibl.weights = weights
        # ----------------------------------------------

        # Predict on the ORIGINAL test data
        predictions, time_per_instance = kibl.predict(X_test, y_test)

        # Compute metrics
        accuracy = float(np.mean(predictions == y_test))
        storage_pct = kibl.get_storage_percentage()

        return {
            'Accuracy': accuracy,
            'Time_per_instance_ms': time_per_instance * 1000.0,
            'Storage_percent': storage_pct,
            'Train_size': original_train_size,
            'Test_size': len(X_test),
            'Instance_base_final_size': len(kibl.instance_base)
        }


    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Runs the feature weighting experiments across all datasets and folds."""
        all_results: List[Dict[str, Any]] = []

        for dataset_name, config in BEST_CONFIGS.items():
            print(f"\n{'=' * 80}")
            print(f"[FW Runner] Processing Dataset: {dataset_name.upper()}")
            print(f"  Using Baseline Config: k={config['k']}, "
                  f"{config['distance']}, {config['voting']}, {config['retention']}")
            print(f"{'=' * 80}\n")

            # Load all 10 folds for this dataset
            print(f"  Loading all 10 folds for '{dataset_name}'...")
            folds = load_all_folds(dataset_name, self.data_dir)
            print("  ...folds loaded.")

            dist_func = self.distance_functions[config['distance']]
            vote_func = self.voting_functions[config['voting']]

            for fold_idx in range(len(folds)):
                print(f"\n    --- Fold {fold_idx} ---")
                X_train, y_train, X_test, y_test = folds[fold_idx]

                for method_name in self.weighting_methods:
                    print(f"    [Method: {method_name}]")

                    # 1. Compute feature weights
                    weights = compute_feature_weights(
                        X_train, y_train,
                        method=method_name,
                        run_mode=RUN_MODE,
                        subsample_size=SUBSAMPLE_SIZE,
                        relieff_neighbors=RELIEFF_NEIGHBORS
                    )

                    # 2. Evaluate k-IBL with these weights
                    metrics = self._evaluate_fold(
                        (X_train, y_train, X_test, y_test),
                        config,
                        dist_func,
                        vote_func,
                        weights
                    )

                    print(f"      → Fold {fold_idx} Metrics: "
                          f"Acc={metrics['Accuracy']:.4f}, "
                          f"Time={metrics['Time_per_instance_ms']:.2f}ms, "
                          f"Storage={metrics['Storage_percent']:.1f}%")

                    # 3. Store result
                    result = {
                        'Dataset': dataset_name,
                        'Fold': fold_idx,
                        'K': config['k'],
                        'Similarity': config['distance'],
                        'Voting': config['voting'],
                        'Retention': config['retention'], # Store baseline retention
                        'Weight_Method': method_name,
                        'Accuracy': metrics['Accuracy'],
                        'Time_per_instance_ms': metrics['Time_per_instance_ms'],
                        'Storage_percent': metrics['Storage_percent'],
                        'Train_size': metrics['Train_size'],
                        'Test_size': metrics['Test_size'],
                        'Instance_base_final_size': metrics['Instance_base_final_size']
                    }
                    all_results.append(result)

        # --- Save final results ---
        fold_results_df = pd.DataFrame(all_results)

        # Compute summary
        summary_df = fold_results_df.groupby(['Dataset', 'Weight_Method']).agg({
            'Accuracy': ['mean', 'std'],
            'Time_per_instance_ms': ['mean', 'std'],
            'Storage_percent': ['mean', 'std']
        }).reset_index()

        summary_df.columns = [
            'Dataset', 'Weight_Method',
            'Mean_Accuracy', 'Std_Accuracy',
            'Mean_Time_ms', 'Std_Time_ms',
            'Mean_Storage_pct', 'Std_Storage_pct'
        ]

        # Save CSVs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fold_file = RESULTS_DIR / f'fw_detailed_fold_results_{timestamp}.csv'
        summary_file = RESULTS_DIR / f'fw_aggregated_results_{timestamp}.csv'

        fold_results_df.to_csv(fold_file, index=False)
        summary_df.to_csv(summary_file, index=False)

        print(f"\n[FW Runner] Saved fold-level results to: {fold_file.relative_to(PROJECT_ROOT)}")
        print(f"[FW Runner] Saved summary results to: {summary_file.relative_to(PROJECT_ROOT)}\n")

        return fold_results_df, summary_df


# --- Main execution ---
if __name__ == "__main__":
    print("\n--- k-IBL Feature Weighting (FW) Experiment Runner (Step 5) ---")

    if not DATA_DIR.is_dir():
        print(f"Error: Data directory not found at {DATA_DIR}")
        print("Expected structure: ./data/datasetsCBR/adult/...")
        sys.exit(1)

    try:
        runner = FeatureWeightingExperimentRunner(DATA_DIR, RESULTS_DIR)
        runner.run()
        print("\n[FW Runner] All FW experiments completed successfully.")
    except KeyboardInterrupt:
        print(f"\n[FW Runner] Experiment interrupted by user.")
        print(f"Partial results may be saved in '{RESULTS_DIR}'.")
    except Exception as e:
        print(f"\n[FW Runner] An error occurred: {e}")
        import traceback
        traceback.print_exc()
