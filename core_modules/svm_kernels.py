"""
This script implements the Support Vector Machine (SVM) part of the
assignment. It uses the scikit-learn library to:
- Define an `svmAlgorithm` function as required.
- Run 10-fold cross-validation for a given dataset and kernel.
- Compare the performance (accuracy, time) of two selected kernels
  (e.g., 'rbf' and 'linear').
"""

import numpy as np
from sklearn.svm import SVC
import time
import os
import sys
from typing import Dict, Any, List, Tuple
from pathlib import Path

# --- Import the pre-processing pipeline ---
try:
    # We assume the parser is named 'parser.py' and is in the same 'src' dir
    from core_modules.parser import preprocess_fold, load_all_folds
except ImportError:
    print("Error: Could not import from 'parser.py'.")
    print("Please ensure 'parser.py' is in the same directory.")
    sys.exit(1)

# --- Type Aliases ---
ProcessedFoldData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
SVMResult = Dict[str, Any]
FoldResults = Dict[str, Any]


def svmAlgorithm(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale'
) -> SVMResult:
    """
    Trains and evaluates an SVM classifier on a single fold.

    This function uses scikit-learn's SVC and times the
    training and testing (prediction) phases separately to
    evaluate efficiency.
    """
    # Initialize the SVM classifier from scikit-learn
    # We allow kernel, C, and gamma to be set for parameter analysis.
    clf = SVC(kernel=kernel, C=C, gamma=gamma, cache_size=500) # Added cache_size for potential speedup

    # --- Training Phase ---
    start_train = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_train

    # --- Testing (Prediction) Phase ---
    start_test = time.time()
    predictions = clf.predict(X_test)
    test_time = time.time() - start_test

    # --- Performance Metrics ---
    accuracy = np.mean(predictions == y_test)

    # Get the number of support vectors (useful for analysis)
    n_support_vectors = len(clf.support_)
    sv_ratio = n_support_vectors / len(X_train) if len(X_train) > 0 else 0

    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'train_time': train_time,
        'test_time': test_time,
        'total_time': train_time + test_time,
        'n_support_vectors': n_support_vectors,
        'support_vector_ratio': sv_ratio
    }


def run_svm_10fold(
        dataset_name: str,
        data_directory: str,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale'
) -> FoldResults:
    """
    Runs the SVM algorithm over all 10 folds for a given dataset
    and kernel configuration.

    Aggregates and prints the mean/std for accuracy and time.
    """
    print(f"\n[SVM Runner] Running SVM (kernel='{kernel}', C={C}, gamma='{gamma}') on '{dataset_name}'")

    # Use the main parser to load all 10 folds at once
    try:
        all_folds: List[ProcessedFoldData] = load_all_folds(dataset_name, data_directory)
    except FileNotFoundError as e:
        print(f"[SVM Runner] Error loading data: {e}")
        return {} # Return empty results

    if not all_folds:
        print("[SVM Runner] No folds were loaded. Aborting SVM run.")
        return {}

    # Store results from each fold
    fold_accuracies: List[float] = []
    fold_train_times: List[float] = []
    fold_test_times: List[float] = []
    fold_total_times: List[float] = []
    fold_n_svs: List[int] = []
    fold_sv_ratios: List[float] = []

    for fold_num, (X_train, y_train, X_test, y_test) in enumerate(all_folds):
        print(f"  Processing Fold {fold_num+1}/10...", end=' ', flush=True)

        try:
            # 1. Run the SVM algorithm
            result = svmAlgorithm(
                X_train, y_train, X_test, y_test,
                kernel=kernel, C=C, gamma=gamma
            )

            # 2. Store results
            fold_accuracies.append(result['accuracy'])
            fold_train_times.append(result['train_time'])
            fold_test_times.append(result['test_time'])
            fold_total_times.append(result['total_time'])
            fold_n_svs.append(result['n_support_vectors'])
            fold_sv_ratios.append(result['support_vector_ratio'])

            # Print statement for fold-by-fold results
            print(f"Acc: {result['accuracy']:.4f}, "
                  f"Time: {result['total_time']:.2f}s, "
                  f"SVs: {result['n_support_vectors']}")

        except Exception as e:
            print(f"Error on fold {fold_num+1}: {e}")
            fold_accuracies.append(np.nan)
            fold_total_times.append(np.nan)
            # ... etc.

    # 3. Aggregate results
    mean_accuracy = np.nanmean(fold_accuracies)
    std_accuracy = np.nanstd(fold_accuracies)
    mean_time = np.nanmean(fold_total_times)
    std_time = np.nanstd(fold_total_times)
    mean_svs = np.nanmean(fold_n_svs)

    # --- Print Summary (for reporting) ---
    print(f"\n{'-' * 80}")
    print(f"SUMMARY FOR {kernel.upper()} KERNEL ('{dataset_name}'):")
    print(f"  Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"  Mean Time:     {mean_time:.3f}s ± {std_time:.3f}s")
    print(f"  Mean Support Vectors: {mean_svs:.1f}")
    print(f"\n  Individual Fold Accuracies:")
    for i, acc in enumerate(fold_accuracies, 1):
        print(f"    Fold {i:2d}: {acc:.4f}")
    print(f"{'-' * 80}")

    # Return all results for further statistical comparison
    return {
        'accuracies': fold_accuracies,
        'train_times': fold_train_times,
        'test_times': fold_test_times,
        'total_times': fold_total_times,
        'n_support_vectors': fold_n_svs,
        'support_vector_ratios': fold_sv_ratios,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_time': mean_time,
        'std_time': std_time
    }


def compare_kernels(dataset_name: str, data_directory: str) -> Dict[str, FoldResults]:
    """
    Compares two selected kernels (RBF and Linear)
    on a given dataset.
    """
    results: Dict[str, FoldResults] = {}

    # Justification: We select 'rbf' and 'linear' as our two kernels.
    # 'rbf' is a powerful, non-linear default.
    # 'linear' is highly efficient and effective on linearly separable data.

    # --- Kernel 1: RBF (Radial Basis Function) ---
    # Default C=1.0 and gamma='scale' are good starting points.
    results['rbf'] = run_svm_10fold(
        dataset_name, data_directory, kernel='rbf', C=1.0, gamma='scale'
    )

    # --- Kernel 2: Linear ---
    # C=1.0 is a standard default.
    results['linear'] = run_svm_10fold(
        dataset_name, data_directory, kernel='linear', C=1.0
    )

    # --- Print final comparison (for quick analysis) ---
    print(f"\n[SVM Runner] FINAL KERNEL COMPARISON ('{dataset_name}'):")
    print(f"  RBF Kernel:    {results['rbf']['mean_accuracy']:.4f} "
          f"± {results['rbf']['std_accuracy']:.4f}")
    print(f"  Linear Kernel: {results['linear']['mean_accuracy']:.4f} "
          f"± {results['linear']['std_accuracy']:.4f}")
    print(f"{'-' * 80}")

    return results


if __name__ == "__main__":

    # --- Configuration ---
    # Use a relative path for the data directory.
    # This assumes the script is run from the project's root directory
    # and the data is in 'Work2_Project/data/datasetsCBR/'.

    # Get the project root
    try:
        PROJECT_ROOT = Path(__file__).resolve().parent
    except NameError:
        PROJECT_ROOT = Path.cwd()

    # Set the data directory path
    DATA_DIR = PROJECT_ROOT / "data" / "datasetsCBR"

    # Select one of the datasets provided.
    DATASET = "pen-based"  # 'pen-based' or 'adult'
    # ---------------------

    print("\n--- SVM Experiment Runner (Test) ---")
    print(f"Dataset: {DATASET}")
    print(f"Data Dir: {DATA_DIR}")

    if not os.path.isdir(DATA_DIR):
        print(f"\nError: Data directory not found at {DATA_DIR}")
        print("Please ensure the 'data/datasetsCBR' folder exists in the project root,")
        print(f"or update the DATA_DIR variable in the __main__ block.")
        sys.exit(1)

    try:
        # Run both kernels and print results
        results = compare_kernels(DATASET, str(DATA_DIR))
        print("\n[SVM Runner] Test run completed successfully.")

    except FileNotFoundError as e:
        print(f"\n[SVM Runner] Error: File not found.")
        print(f"  Details: {e}")
    except ImportError as e:
        print(f"\n[SVM Runner] Error: Could not import module.")
        print(f"  Details: {e}")
    except Exception as e:
        print(f"\n[SVM Runner] An unexpected error occurred:")
        print(f"  Details: {e}")