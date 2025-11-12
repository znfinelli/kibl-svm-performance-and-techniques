"""
This script runs the Instance Reduction (IR) experiments (Step 6).
It implements the `ir_KIBLAlgorithm` as required by the assignment.

It does the following:
1.  Loads the "best" k-IBL configuration for each dataset.
2.  For each fold of a dataset:
    a. Applies each of the 3 IR algorithms (ENN, TCNN, ICF) from
       `ir_reducers.py` to the *training set*.
    b. Saves the reduced training set as .npy files (for the SVM
       analysis, as required by Step 6.e).
    c. Trains the best k-IBL algorithm on this *reduced* set.
    d. Tests the trained algorithm on the *original* test set.
3.  Saves detailed fold-by-fold results and an aggregated summary,
    comparing performance (Accuracy, Time, Storage).
"""

# --- Path Setup ---
from pathlib import Path
import sys
import os
from typing import Callable, Dict, List, Tuple, Any

# Define paths
try:
    # This works when run as a script
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for interactive/notebook use
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'datasetsCBR'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'ir_baseline'
# This is the directory where the reduced sets will be saved for svm_ir_runner.py
NPY_SAVE_DIR = PROJECT_ROOT / 'results' / 'ir_reduced_datasets'

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NPY_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to path to find other modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- End Path Setup ---

import time
from datetime import datetime
import numpy as np
import pandas as pd

# --- Import Project Modules ---
try:
    from core_modules.parser import load_all_folds
    from core_modules.kibl import kIBLAlgorithm
    from core_modules.distances import euclidean_distance, cosine_distance, heom_distance
    from core_modules.voting import modified_plurality_vote, borda_count_vote
    # Import the reducer functions from ir_reducers.py
    from core_modules.ir_reducers import enn, tcnn, icf
except ImportError as exc:
    print(f"Error: Could not import project modules (parser, kibl, ir_reducers, etc.)")
    print(f"Ensure they are in the project root directory: {PROJECT_ROOT}")
    raise

# --- Type Aliases ---
FoldData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
DataSet = Tuple[np.ndarray, np.ndarray]
KIBLMetrics = Dict[str, Any]
ReducerFunc = Callable[..., DataSet] # Use ... to allow for k param in enn

# --- Experiment Configuration ---

# Best baseline configurations found in the first experiment
# !! UPDATED with your provided winners (pen-based set to NR) !!
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

# Reduction techniques to test
# We wrap them in lambdas to provide a consistent interface
REDUCTION_TECHNIQUES: Dict[str, ReducerFunc] = {
    'ENN': lambda x, y: enn(x, y, k=3), # Use k=3 for ENN
    'TCNN': tcnn,
    'ICF': lambda x, y: icf(x, y, k=1), # Use k=1 for ICF
}

# --- K-IBL Helpers ---
DISTANCE_FUNCS: Dict[str, Callable] = {
    'Euclidean': euclidean_distance,
    'Cosine': cosine_distance,
    'HEOM': heom_distance
}
VOTING_FUNCS: Dict[str, Callable] = {
    'ModPlurality': modified_plurality_vote,
    'BordaCount': borda_count_vote
}


# -----------------------------


def ir_KIBLAlgorithm(
        X_train: np.ndarray,
        y_train: np.ndarray,
        reducer_func: ReducerFunc,
        reducer_name: str
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    This function applies the instance reduction preprocessing.
    It's the core of the `ir_KIBLAlgorithm`.

    Returns:
        - X_reduced: The reduced feature set
        - y_reduced: The reduced label set
        - storage_pct: The percentage of instances remaining
        - reduction_time: The time taken to perform the reduction
    """
    print(f"    Applying {reducer_name}...")
    start_time = time.time()

    # Apply the selected reduction technique (e.g., enn, tcnn, icf)
    X_reduced, y_reduced = reducer_func(X_train, y_train)

    reduction_time = time.time() - start_time

    # Calculate storage as a percentage of the original
    original_size = len(X_train)
    reduced_size = len(X_reduced)
    storage_pct = (reduced_size / original_size) * 100.0 if original_size > 0 else 0

    print(f"    → Reduction took {reduction_time:.2f}s. "
          f"Storage: {reduced_size}/{original_size} ({storage_pct:.1f}%)")

    return X_reduced, y_reduced, storage_pct, reduction_time


def run_ir_experiments():
    """
    Main runner for the Instance Reduction experiments.
    """
    all_results: List[Dict[str, Any]] = []

    for dataset_name, config in BEST_CONFIGS.items():
        print(f"\n{'=' * 80}")
        print(f"[IR Runner] Processing Dataset: {dataset_name.upper()}")
        print(f"  Using Baseline Config: k={config['k']}, "
              f"{config['distance']}, {config['voting']}, {config['retention']}")
        print(f"{'=' * 80}\n")

        # Check if data directory exists
        if not DATA_DIR.is_dir():
            print(f"Error: Data directory not found at {DATA_DIR}")
            print("Expected structure: ./data/datasetsCBR/adult/...")
            continue

        # Load all 10 folds
        print(f"  Loading all 10 folds for '{dataset_name}'...")
        folds = load_all_folds(dataset_name, str(DATA_DIR))
        print("  ...folds loaded.")

        # Get the callable functions from the config names
        dist_func = DISTANCE_FUNCS[config['distance']]
        vote_func = VOTING_FUNCS[config['voting']]

        # Iterate over each reduction technique
        for technique_name, reducer_func in REDUCTION_TECHNIQUES.items():
            print(f"\n  --- Testing IR Technique: {technique_name} ---")

            # Create subdirectories for saving .npy files
            # e.g., results/ir_reduced_datasets/ENN/adult/
            npy_dataset_dir = NPY_SAVE_DIR / technique_name / dataset_name
            npy_dataset_dir.mkdir(parents=True, exist_ok=True)
            print(f"    (Saving reduced .npy files to: {npy_dataset_dir.relative_to(PROJECT_ROOT)})")

            # Iterate over all 10 folds
            for fold_idx in range(len(folds)):
                print(f"\n    --- Fold {fold_idx} ({technique_name}) ---")
                X_train_orig, y_train_orig, X_test, y_test = folds[fold_idx]

                # 1. Apply Instance Reduction
                # This is the "ir_KIBLAlgorithm" preprocessing step
                X_train_reduced, y_train_reduced, storage_pct, reduction_time = ir_KIBLAlgorithm(
                    X_train_orig, y_train_orig, reducer_func, technique_name
                )

                # 2. Save the reduced training set (for SVM analysis )
                npy_X_path = npy_dataset_dir / f"fold_{fold_idx}_X.npy"
                npy_y_path = npy_dataset_dir / f"fold_{fold_idx}_y.npy"
                np.save(npy_X_path, X_train_reduced)
                np.save(npy_y_path, y_train_reduced)

                # 3. Evaluate k-IBL on the *reduced* set
                # Instantiate k-IBL with the baseline config
                kibl = kIBLAlgorithm(
                    k=config['k'],
                    distance_func=dist_func,
                    voting_func=vote_func,
                    retention_policy=config['retention']
                )

                # Fit on the REDUCED data
                train_start = time.time()
                kibl.fit(X_train_reduced, y_train_reduced)
                train_time = time.time() - train_start

                # Predict on the ORIGINAL test data
                predictions, time_per_instance = kibl.predict(X_test, y_test)

                # 4. Compute metrics
                accuracy = float(np.mean(predictions == y_test))
                # Efficiency = reduction_time + train_time + prediction_time
                total_time = reduction_time + train_time + (time_per_instance * len(X_test))

                print(f"      → Fold {fold_idx} Metrics: "
                      f"Acc={accuracy:.4f}, "
                      f"Total Time={total_time:.2f}s, "
                      f"Storage={storage_pct:.1f}%")

                # 5. Store result
                result = {
                    'Dataset': dataset_name,
                    'Fold': fold_idx,
                    'K': config['k'],
                    'Similarity': config['distance'],
                    'Voting': config['voting'],
                    'Retention': config['retention'],  # Baseline retention policy
                    'IR_Technique': technique_name,
                    'Accuracy': accuracy,
                    'Time_per_instance_ms': time_per_instance * 1000.0,
                    'Reduction_Time_s': reduction_time,
                    'Train_Time_s': train_time,
                    'Total_Time_s': total_time,
                    'Storage_percent': storage_pct,
                    'Original_Train_Size': len(X_train_orig),
                    'Reduced_Train_Size': len(X_train_reduced)
                }
                all_results.append(result)

    # --- Save final results ---
    fold_results_df = pd.DataFrame(all_results)

    # Compute summary
    summary_df = fold_results_df.groupby(['Dataset', 'IR_Technique']).agg({
        'Accuracy': ['mean', 'std'],
        'Total_Time_s': ['mean', 'std'],
        'Storage_percent': ['mean', 'std']
    }).reset_index()

    summary_df.columns = [
        'Dataset', 'IR_Technique',
        'Mean_Accuracy', 'Std_Accuracy',
        'Mean_Time_s', 'Std_Time_s',
        'Mean_Storage_pct', 'Std_Storage_pct'
    ]

    # Save CSVs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fold_file = RESULTS_DIR / f'ir_detailed_fold_results_{timestamp}.csv'
    summary_file = RESULTS_DIR / f'ir_aggregated_results_{timestamp}.csv'

    fold_results_df.to_csv(fold_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    print(f"\n[IR Runner] Saved fold-level results to: {fold_file.relative_to(PROJECT_ROOT)}")
    print(f"[IR Runner] Saved summary results to: {summary_file.relative_to(PROJECT_ROOT)}\n")

    # Display summary
    print(f"{'-' * 80}")
    print("Instance Reduction Experiment Summary (mean ± std)")
    print(f"{'-' * 80}")
    print(f"  {'Dataset':<12} | {'IR_Technique':<6} | "
          f"{'Accuracy':<18} | {'Time (s)':<18} | {'Storage %':<18}")
    print(f"  {'-' * 80}")
    for _, row in summary_df.iterrows():
        acc_str = f"{row['Mean_Accuracy']:.4f}±{row['Std_Accuracy']:.4f}"
        time_str = f"{row['Mean_Time_s']:.2f}±{row['Std_Time_s']:.2f}"
        storage_str = f"{row['Mean_Storage_pct']:.1f}±{row['Std_Storage_pct']:.1f}"
        print(f"  {row['Dataset'].upper():<12} | {row['IR_Technique']:<6} | "
              f"{acc_str:<18} | {time_str:<18} | {storage_str:<18}")
    print(f"{'-' * 80}\n")

    return fold_results_df, summary_df


# --- Main execution ---
if __name__ == "__main__":
    print("\n--- k-IBL Instance Reduction (IR) Experiment Runner (Step 6) ---")
    try:
        run_ir_experiments()
        print("\n[IR Runner] All IR experiments completed successfully.")
    except KeyboardInterrupt:
        print(f"\n[IR Runner] Experiment interrupted by user.")
        print(f"Partial results may be saved in '{RESULTS_DIR}'.")
    except Exception as e:
        print(f"\n[IR Runner] An error occurred: {e}")
        import traceback

        traceback.print_exc()