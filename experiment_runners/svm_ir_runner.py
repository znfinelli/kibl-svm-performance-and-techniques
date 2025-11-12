"""
This script analyzes how the instance reduction of the training set affects
the results obtained by the svm algorithm (Step 6.e).

It does the following:
1.  Loads the "best" SVM config from the baseline tuning.
2.  Loads the original 10-fold datasets.
3.  Loads the pre-reduced .npy training sets.
4.  Runs the best SVM on both the 'Baseline' (full) training set
    and the three 'IR' (reduced) training sets (ENN, TCNN, ICF).
5.  Saves a detailed JSON file containing all fold-level results
    (accuracies, times, svs) for the stats script to analyze.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from scipy import stats
import time
import os
import sys
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime

# --- Path Setup ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- End Path Setup ---

# --- Import Project Modules ---
try:
    from core_modules.parser import load_all_folds
    from core_modules.svm_kernels import svmAlgorithm
except ImportError:
    print(f"Error: Could not import project modules (parser, svm_kernels).")
    print(f"Ensure they are in the project root directory: {PROJECT_ROOT}")
    sys.exit(1)
# -----------------------------

# --- Type Aliases ---
ProcessedFoldData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
SVMResult = Dict[str, Any]
FoldResults = Dict[str, Any]


def load_reduced_fold(
        dataset_name: str,
        fold_number: int, # 1-based index (1-10)
        reduction_technique: str,
        reduction_base_path: Path,
        original_test_fold: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a training set that was reduced by an IR algorithm
    and pairs it with the corresponding *original* test set.
    """
    npy_fold_idx = fold_number - 1
    reduction_folder = reduction_base_path / reduction_technique / dataset_name
    npy_X_path = reduction_folder / f"fold_{npy_fold_idx}_X.npy"
    npy_y_path = reduction_folder / f"fold_{npy_fold_idx}_y.npy"

    if not npy_X_path.exists():
        raise FileNotFoundError(f"Reduced training file not found: {npy_X_path}")
    if not npy_y_path.exists():
        raise FileNotFoundError(f"Reduced training label file not found: {npy_y_path}")

    X_train = np.load(npy_X_path)
    y_train = np.load(npy_y_path)
    X_test, y_test = original_test_fold

    return X_train, y_train, X_test, y_test


def run_svm_with_reduction(
        dataset_name: str,
        reduction_technique: str, # 'ENN', 'TCNN', 'ICF', or 'Baseline'
        reduction_base_path: Path,
        original_folds: List[ProcessedFoldData],
        svm_params: Dict[str, Any]
) -> FoldResults:
    """
    Runs SVM over all 10 folds using a specific IR-reduced training set.
    If 'reduction_technique' is 'Baseline', it uses the original training set.
    """
    print(f"\n[SVM-IR] Testing: {dataset_name.upper()} | {reduction_technique} | "
          f"kernel={svm_params['kernel']}, C={svm_params['C']}, gamma={svm_params['gamma']}")

    fold_accuracies: List[float] = []
    fold_times: List[float] = []
    fold_support_vectors: List[int] = []
    fold_train_sizes: List[int] = []

    for fold_num in range(1, 11): # 1-based index
        print(f"  Processing Fold {fold_num}/10... ", end='', flush=True)

        try:
            X_train_orig, y_train_orig, X_test, y_test = original_folds[fold_num-1]

            if reduction_technique == 'Baseline':
                X_train, y_train = X_train_orig, y_train_orig
            else:
                X_train, y_train, _, _ = load_reduced_fold(
                    dataset_name, fold_num, reduction_technique,
                    reduction_base_path, (X_test, y_test)
                )

            fold_train_sizes.append(len(X_train))

            if len(X_train) == 0:
                print("Acc: 0.0000 (Empty Train Set), Time: 0.00s, SV: 0")
                fold_accuracies.append(0.0)
                fold_times.append(0.0)
                fold_support_vectors.append(0)
                continue

            result = svmAlgorithm(
                X_train, y_train, X_test, y_test,
                kernel=svm_params['kernel'],
                C=svm_params['C'],
                gamma=svm_params['gamma']
            )

            fold_accuracies.append(result['accuracy'])
            fold_times.append(result['total_time'])
            fold_support_vectors.append(result['n_support_vectors'])

            print(f"Acc: {result['accuracy']:.4f}, "
                  f"Time: {result['total_time']:.2f}s, "
                  f"SV: {result['n_support_vectors']}")

        except FileNotFoundError as e:
            print(f"File not found for fold {fold_num}. Skipping.")
            fold_accuracies.append(np.nan)
            fold_times.append(np.nan)
            fold_support_vectors.append(np.nan)
        except Exception as e:
            print(f"Error processing fold {fold_num}: {e}")
            fold_accuracies.append(np.nan)
            fold_times.append(np.nan)
            fold_support_vectors.append(np.nan)

    mean_accuracy = np.nanmean(fold_accuracies)
    std_accuracy = np.nanstd(fold_accuracies)
    mean_time = np.nanmean(fold_times)
    std_time = np.nanstd(fold_times)
    mean_sv = np.nanmean(fold_support_vectors)
    mean_train_size = np.nanmean(fold_train_sizes)

    print(f"  → Summary: mean_acc={mean_accuracy:.4f} ± {std_accuracy:.4f}, "
          f"mean_time={mean_time:.3f}s ± {std_time:.3f}s, "
          f"mean_SV≈{mean_sv:.0f}, mean_train_size≈{mean_train_size:.0f}")

    return {
        'accuracies': fold_accuracies,
        'times': fold_times,
        'support_vectors': fold_support_vectors,
        'train_sizes': fold_train_sizes,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_time': mean_time,
        'std_time': std_time,
        'mean_sv': mean_sv,
        'mean_train_size': mean_train_size
    }


if __name__ == "__main__":

    # --- Configuration ---
    DATA_DIR = PROJECT_ROOT / "data" / "datasetsCBR"
    REDUCTION_BASE = PROJECT_ROOT / "results" / "ir_reduced_datasets"
    OUTPUT_DIR = PROJECT_ROOT / "results" / "svm_ir_analysis"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # ---------------------

    DATASETS = ['adult', 'pen-based']
    TECHNIQUES = ['Baseline', 'TCNN', 'ICF', 'ENN'] # 'Baseline' is the full dataset

    # Best SVM parameters
    # !! UPDATE THESE with your Step 4 analysis  results !!
    SVM_PARAMS = {
        'adult': {'kernel': 'rbf', 'C': 10.0, 'gamma': 1.0},
        'pen-based': {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'}
    }

    if not DATA_DIR.is_dir():
        print(f"Error: Data directory not found at {DATA_DIR}")
        sys.exit(1)
    if not REDUCTION_BASE.is_dir():
        print(f"Error: Reduced dataset directory not found at {REDUCTION_BASE}")
        print("Please run 'ir_baseline_runner.py' first to generate the .npy files.")
        sys.exit(1)

    print("\n--- SVM with Instance Reduction Experiment (Step 6.e) ---")

    # This dictionary will store all fold-level results for the stats script
    all_json_results: Dict[str, Any] = {'datasets': {}}

    for dataset in DATASETS:
        print(f"\n[ Processing Dataset: {dataset.upper()} ]")

        print("  Loading original 10-fold data...")
        original_folds = load_all_folds(dataset, str(DATA_DIR))
        print("  ...data loaded.")

        params = SVM_PARAMS[dataset]
        dataset_results: Dict[str, Any] = {}

        for technique in TECHNIQUES:
            try:
                results = run_svm_with_reduction(
                    dataset_name=dataset,
                    reduction_technique=technique,
                    reduction_base_path=REDUCTION_BASE,
                    original_folds=original_folds,
                    svm_params=params
                )
                dataset_results[technique] = results

            except FileNotFoundError as e:
                print(f"\n  ERROR: Could not find reduced data for {technique}.")
                print(f"    Skipping {technique} for {dataset}\n")
                continue
            except Exception as e:
                print(f"\n  ERROR: An unexpected error occurred with {technique}.")
                continue

        all_json_results['datasets'][dataset] = dataset_results

    # --- Save detailed fold-level results to JSON ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = OUTPUT_DIR / f"svm_ir_fold_results_{timestamp}.json"

    try:
        with open(json_file, 'w') as f:
            json.dump(all_json_results, f, indent=4)
        print(f"\n[SVM-IR] Detailed fold-level results saved to: {json_file.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print(f"\n[SVM-IR] Error saving results to JSON: {e}")

    print(f"\n[SVM-IR] All experiments complete.")
    print("Next, run 'svm_irSVM_stats_runner.py' to analyze these results.")