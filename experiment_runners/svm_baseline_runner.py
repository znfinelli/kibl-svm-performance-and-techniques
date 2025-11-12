"""
This is the EXPERIMENT RUNNER script for analyzing the SVM, with different
parameter combinations (Step 4).

It does the following:
1.  Defines a grid of SVM parameters to test (kernels, C, gamma).
2.  For each dataset and each parameter combination:
    a. Runs the `svmAlgorithm` over all 10 folds.
    b. Calculates the mean accuracy and time.
3.  Saves all fold-by-fold results and a summary to a .json file.
4.  This .json file can be used for statistical analysis.
"""

import numpy as np
import pandas as pd
import time
import os
import json
import itertools
from typing import Dict, Any, List
from pathlib import Path

# --- Path Setup ---
import sys

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

# --- Import Project Modules ---
try:
    # We need the preprocessor from parser.py
    from core_modules.parser import load_all_folds
    # We use the svmAlgorithm function from svm_kernels.py
    from core_modules.svm_kernels import svmAlgorithm
except ImportError as e:
    print(f"Error: Could not import project modules (parser, svm_kernels). {e}")
    print(f"Ensure they are in the project root directory: {PROJECT_ROOT}")
    sys.exit(1)
# -----------------------------

# --- Configuration ---
DATA_DIR = PROJECT_ROOT / "data" / "datasetsCBR"
OUTPUT_DIR = PROJECT_ROOT / "results" / "svm_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['adult', 'pen-based']

# Define the parameter grid to search
# This grid is a good starting point for analysis
PARAM_GRID = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 1.0] # gamma is ignored by 'linear' kernel
}
# ---------------------

def run_svm_tuning_for_dataset(dataset_name: str, all_folds: List) -> Dict[str, Any]:
    """
    Runs the full parameter grid search for a single dataset.
    """
    print(f"\n[SVM Tune] Starting tuning for: {dataset_name.upper()}")

    # Generate all unique combinations of parameters
    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Filter out invalid combinations
    # (gamma is only used for 'rbf' kernel)
    valid_params = []
    for params in param_combinations:
        if params['kernel'] == 'linear':
            # Remove gamma, set to 'scale' for clarity
            params['gamma'] = 'scale'
            if params not in valid_params:
                valid_params.append(params)
        else:
            valid_params.append(params)

    num_configs = len(valid_params)
    print(f"  Testing {num_configs} unique parameter configurations...")

    # Store results:
    # accuracy_matrix: (n_folds x n_configs) - for stats
    # time_matrix: (n_folds x n_configs) - for stats
    # summary: Dict[config_name, metrics] - for simple lookup
    accuracy_matrix = np.zeros((len(all_folds), num_configs))
    time_matrix = np.zeros((len(all_folds), num_configs))
    summary = {}
    config_names = []  # To map matrix columns to names

    # Loop 1: Configurations
    for i, params in enumerate(valid_params):
        config_name = f"k={params['kernel']},C={params['C']},g={params['gamma']}"
        config_names.append(config_name)
        print(f"  Config {i + 1}/{num_configs}: {config_name}")

        fold_accuracies = []
        fold_times = []

        # Loop 2: Folds
        for fold_idx in range(len(all_folds)):
            X_train, y_train, X_test, y_test = all_folds[fold_idx]

            # Run the SVM algorithm
            result = svmAlgorithm(
                X_train, y_train, X_test, y_test,
                kernel=params['kernel'],
                C=params['C'],
                gamma=params['gamma']
            )

            # Store fold results
            fold_accuracies.append(result['accuracy'])
            fold_times.append(result['total_time'])

            # Update matrices
            accuracy_matrix[fold_idx, i] = result['accuracy']
            time_matrix[fold_idx, i] = result['total_time']

        # Store summary stats for this config
        summary[config_name] = {
            'params': params,
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
            'mean_time': np.mean(fold_times),
            'std_time': np.std(fold_times),
        }
        print(f"    → Mean Acc: {summary[config_name]['mean_accuracy']:.4f}, "
              f"Mean Time: {summary[config_name]['mean_time']:.3f}s")

    # --- Save results to JSON ---
    output_data = {
        'dataset': dataset_name,
        'config_names': config_names,
        'param_grid': PARAM_GRID,
        'accuracy_matrix': accuracy_matrix.tolist(),  # Convert np.array to list
        'time_matrix': time_matrix.tolist(),
        'summary': summary
    }

    json_filename = OUTPUT_DIR / f"svm_params_{dataset_name}_results.json"
    with open(json_filename, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"  [SVM Tune] Saved results to: {json_filename.relative_to(PROJECT_ROOT)}")
    return output_data


def main():
    """
    Main function to run SVM parameter tuning for all datasets.
    """
    print("\n--- SVM Parameter Tuning Experiment Runner (Step 4) ---")

    # Check if data directory exists
    if not DATA_DIR.is_dir():
        print(f"Error: Data directory not found at {DATA_DIR}")
        print("Expected structure: ./data/datasetsCBR/adult/...")
        return

    all_results = {}

    for dataset in DATASETS:
        # Load data once per dataset
        print(f"\n[SVM Tune] Loading data for {dataset}...")
        try:
            all_folds = load_all_folds(dataset_name=dataset, data_directory=str(DATA_DIR))
        except FileNotFoundError as e:
            print(f"  Error: Data not found for {dataset}. {e}")
            continue
        except Exception as e:
            print(f"  Error loading folds for {dataset}. {e}")
            continue
        print(f"  ...loaded {len(all_folds)} folds.")

        # Run the tuning
        dataset_results = run_svm_tuning_for_dataset(dataset, all_folds)
        all_results[dataset] = dataset_results

    print("\n[SVM Tune] All tuning experiments complete.")


if __name__ == "__main__":
    main()