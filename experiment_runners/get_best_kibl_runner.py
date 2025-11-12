"""
This script orchestrates the k-IBL baseline analysis (Step 3).
It iterates through:
- Multiple datasets
- All 10 folds for each dataset
- All required combinations of K, distance, voting, and retention.

It calculates performance metrics (accuracy, time, storage)
for each run, saves detailed fold-level results, and then aggregates
these results to find the "best" k-IBL configuration.
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import sys
from typing import List, Dict, Any, Callable, Tuple
from pathlib import Path

# --- Path Setup ---
try:
    # This works when run as a script
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for interactive/notebook use
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
# Add project root to path to find other modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- End Path Setup ---

# --- Import Project Modules ---
try:
    # Use the corrected module names
    from core_modules.parser import load_all_folds
    from core_modules.kibl import kIBLAlgorithm
    from core_modules.distances import euclidean_distance, cosine_distance, heom_distance
    from core_modules.voting import modified_plurality_vote, borda_count_vote
except ImportError:
    print("Error: Could not import local modules (parser, kibl, etc.)")
    print(f"Ensure files are in the correct directory: {PROJECT_ROOT}")
    sys.exit(1)
# -----------------------------


class ExperimentRunner:
    """
    Manages the setup, execution, and reporting of all k-IBL experiments.
    """

    def __init__(self, data_dir: Path, output_dir: Path):
        """
        Initializes the experiment runner with all parameters.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Create results directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Runner] Created output directory: {output_dir.relative_to(PROJECT_ROOT)}")

        # --- Experiment Parameters (as required by assignment) ---

        # Set to [3, 5, 7] as required by the assignment
        self.k_values: List[int] = [3, 5, 7]

        # Using two datasets as required
        self.datasets: List[str] = ['adult', 'pen-based']

        # Three distance functions
        self.distance_functions: Dict[str, Callable] = {
            'Euclidean': euclidean_distance,
            'Cosine': cosine_distance,
            'HEOM': heom_distance
        }

        # Two voting policies
        self.voting_functions: Dict[str, Callable] = {
            'ModPlurality': modified_plurality_vote,
            'BordaCount': borda_count_vote
        }

        # Four retention policies
        self.retention_policies: List[str] = ['NR', 'AR', 'DC', 'DD']
        # --------------------------------------------------------

        # Calculate total runs for progress tracking
        self.configs_per_dataset = (
                len(self.k_values) *
                len(self.distance_functions) *
                len(self.voting_functions) *
                len(self.retention_policies)
        )
        self.total_configurations = self.configs_per_dataset * len(self.datasets)
        self.total_fold_runs = self.total_configurations * 10  # 10 folds each

        print("\n[Runner] Experiment setup:")
        print(f"  K values: {self.k_values}")
        print(f"  Datasets: {self.datasets}")
        print(f"  Distance functions: {list(self.distance_functions.keys())}")
        print(f"  Voting schemes: {list(self.voting_functions.keys())}")
        print(f"  Retention policies: {self.retention_policies}")
        print(f"  Total configurations: {self.total_configurations} ({self.configs_per_dataset} per dataset)")
        print(f"  Total runs: {self.total_fold_runs} (10 folds each)")

    def run_all_experiments(self) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Executes the main experiment loop.
        """
        print("\n[Runner] Starting experiments...")
        overall_start_time = time.time()

        # Storage for all individual fold results
        all_fold_results: List[Dict[str, Any]] = []

        # Progress tracking
        fold_run_count = 0

        # Initial estimate per run (will be updated)
        estimated_seconds_per_run = 10.0

        # Main experiment loops
        for k in self.k_values:
            print(f"\n--- Processing K = {k} ---")

            for dataset_name in self.datasets:
                print(f"\n  [Dataset: {dataset_name}]")

                # Load all 10 folds for this dataset (done once per dataset)
                load_start = time.time()
                folds = load_all_folds(dataset_name, str(self.data_dir))
                load_time = time.time() - load_start
                print(f"    Loaded {len(folds)} folds ({load_time:.1f}s)")

                # Iterate through all configuration combinations
                for dist_name, dist_func in self.distance_functions.items():
                    for vote_name, vote_func in self.voting_functions.items():
                        for retention in self.retention_policies:

                            config_name = f"K={k}, {dist_name}, {vote_name}, {retention}"
                            print(f"\n    [Config: {config_name}]")

                            # Run on all 10 folds
                            for fold_idx in range(10):
                                fold_start_time = time.time()

                                # Get this fold's data
                                X_train, y_train, X_test, y_test = folds[fold_idx]
                                original_train_size = X_train.shape[0]

                                # 1. Create k-IBL instance
                                kibl = kIBLAlgorithm(
                                    k=k,
                                    distance_func=dist_func,
                                    voting_func=vote_func,
                                    retention_policy=retention
                                )

                                # 2. Train (fit)
                                kibl.fit(X_train, y_train)

                                # 3. Test (predict)
                                # Pass y_test for retention policies
                                predictions, time_per_instance = kibl.predict(X_test, y_test)

                                # 4. Calculate performance metrics
                                accuracy = np.mean(predictions == y_test)
                                storage_pct = kibl.get_storage_percentage() # Use updated method

                                # 5. Store this fold's result
                                fold_result = {
                                    'K': k,
                                    'Dataset': dataset_name,
                                    'Distance': dist_name,
                                    'Voting': vote_name,
                                    'Retention': retention,
                                    'Fold': fold_idx,
                                    'Accuracy': accuracy,
                                    'Time_per_instance_ms': time_per_instance * 1000,
                                    'Storage_percent': storage_pct,
                                    'Train_size': original_train_size,
                                    'Test_size': len(X_test),
                                    'Instance_base_final_size': len(kibl.instance_base)
                                }
                                all_fold_results.append(fold_result)
                                fold_run_count += 1

                                # Update time estimate (exponential moving average)
                                fold_elapsed = time.time() - fold_start_time
                                estimated_seconds_per_run = (
                                        0.9 * estimated_seconds_per_run +
                                        0.1 * fold_elapsed
                                )

                                # Calculate ETA
                                remaining_runs = self.total_fold_runs - fold_run_count
                                eta_seconds = remaining_runs * estimated_seconds_per_run
                                eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                                progress_pct = (fold_run_count / self.total_fold_runs) * 100

                                # Progress display
                                print(f"      Fold {fold_idx}: Acc={accuracy:.4f} | "
                                      f"Time={time_per_instance * 1000:.2f}ms | "
                                      f"Storage={storage_pct:.1f}% | "
                                      f"[{fold_run_count}/{self.total_fold_runs} ({progress_pct:.1f}%) | "
                                      f"ETA: {eta_time.strftime('%H:%M:%S')}]")

                                # Save intermediate results every 20 fold runs
                                if fold_run_count % 20 == 0:
                                    self._save_intermediate_results(all_fold_results, fold_run_count)

        # --- All experiments finished ---
        total_elapsed = time.time() - overall_start_time
        hours, rem = divmod(total_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"\n[Runner] Experiments completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        if fold_run_count > 0:
            print(f"  Total runs: {fold_run_count}, Avg: {total_elapsed / fold_run_count:.1f}s per fold run")

        # Save detailed fold-level results
        fold_results_file = self._save_fold_results(all_fold_results)
        print(f"\n[Runner] Saved detailed results: {fold_results_file.relative_to(PROJECT_ROOT)} ({len(all_fold_results)} rows)")

        # Aggregate and save summary results
        aggregated_results = self._aggregate_by_configuration(all_fold_results)
        agg_results_file = self._save_aggregated_results(aggregated_results)
        print(f"[Runner] Saved aggregated results: {agg_results_file.relative_to(PROJECT_ROOT)} ({len(aggregated_results)} configs)")

        # Generate and display summary
        self._generate_summary(aggregated_results)

        return all_fold_results, aggregated_results

    def _aggregate_by_configuration(self, fold_results: List[Dict]) -> pd.DataFrame:
        """
        Aggregates the 10-fold results for each configuration
        by calculating the mean and std deviation.
        """
        df = pd.DataFrame(fold_results)
        if df.empty:
            return pd.DataFrame()

        # Group by everything except Fold number
        config_columns = ['K', 'Dataset', 'Distance', 'Voting', 'Retention']

        # Calculate mean and std for each metric
        aggregated = df.groupby(config_columns).agg({
            'Accuracy': ['mean', 'std'],
            'Time_per_instance_ms': ['mean', 'std'],
            'Storage_percent': ['mean', 'std']
        }).reset_index()

        # Flatten multi-level column names
        aggregated.columns = [
            'K', 'Dataset', 'Distance', 'Voting', 'Retention',
            'Mean_Accuracy', 'Std_Accuracy',
            'Mean_Time_ms', 'Std_Time_ms',
            'Mean_Storage_pct', 'Std_Storage_pct'
        ]

        # Sort by dataset, then by accuracy (descending)
        aggregated = aggregated.sort_values(
            by=['Dataset', 'Mean_Accuracy'],
            ascending=[True, False]
        )
        return aggregated

    def _save_intermediate_results(self, results: List[Dict], count: int) -> None:
        """Saves a checkpoint of fold results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f'intermediate_folds_kibl_{timestamp}_({count}_runs).csv'
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"      [Checkpoint saved: {filename.relative_to(PROJECT_ROOT)}]")

    def _save_fold_results(self, results: List[Dict]) -> Path:
        """Saves the final, complete list of all fold runs."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f'kibl_detailed_fold_results_FINAL_{timestamp}.csv'
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        return filename

    def _save_aggregated_results(self, aggregated_df: pd.DataFrame) -> Path:
        """Saves the aggregated (mean/std) results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f'kibl_aggregated_configs_FINAL_{timestamp}.csv'
        aggregated_df.to_csv(filename, index=False)
        return filename

    def _generate_summary(self, aggregated_df: pd.DataFrame) -> None:
        """
        Prints a summary of the best results for each dataset.
        """
        print(f"\n[Runner] Results Summary (mean ± std across 10 folds):")

        for dataset in self.datasets:
            dataset_results = aggregated_df[aggregated_df['Dataset'] == dataset].copy()
            if dataset_results.empty:
                print(f"\nNo results found for {dataset.upper()}")
                continue

            print(f"\n--- {dataset.upper()} ({len(dataset_results)} configurations) ---")

            # Table header
            print(f"\n{'K':<3} {'Distance':<11} {'Voting':<14} {'Ret':<4} "
                  f"{'Accuracy (mean±std)':<22} {'Time_ms (mean±std)':<22} {'Storage% (mean±std)':<20}")
            print("-" * 99)

            # Show all configurations
            for _, row in dataset_results.iterrows():
                acc_str = f"{row['Mean_Accuracy']:.4f}±{row['Std_Accuracy']:.4f}"
                time_str = f"{row['Mean_Time_ms']:.2f}±{row['Std_Time_ms']:.2f}"
                storage_str = f"{row['Mean_Storage_pct']:.1f}±{row['Std_Storage_pct']:.1f}"

                print(f"{row['K']:<3} {row['Distance']:<11} {row['Voting']:<14} {row['Retention']:<4} "
                      f"{acc_str:<22} {time_str:<22} {storage_str:<20}")

            # Highlight best configuration
            best_config = dataset_results.iloc[0]  # Already sorted by accuracy
            print("-" * 99)
            print(f"Best for {dataset}:")
            print(f"  Config: K={best_config['K']}, {best_config['Distance']}, "
                  f"{best_config['Voting']}, {best_config['Retention']}")
            print(f"  Accuracy: {best_config['Mean_Accuracy']:.4f} ± {best_config['Std_Accuracy']:.4f}")
            print(f"  Time:     {best_config['Mean_Time_ms']:.2f}ms ± {best_config['Std_Time_ms']:.2f}ms")
            print(f"  Storage:  {best_config['Mean_Storage_pct']:.1f}% ± {best_config['Std_Storage_pct']:.1f}%")


def main():
    """
    Main entry point for running experiments.
    """
    print("\n--- k-IBL Baseline Experiment Runner (Step 3) ---")

    # Use relative paths from the project root
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "results" / "kibl_baseline"

    # Check if data directory exists
    if not data_dir.is_dir():
        print(f"Error: Data directory not found at {data_dir}")
        print("Please place datasets in a 'data/datasetsCBR' folder.")
        print("Expected structure: ./data/datasetsCBR/adult/...")
        return

    # Correct data_dir to point to the 'datasetsCBR' folder
    data_dir = data_dir / "datasetsCBR"
    if not data_dir.is_dir():
        print(f"Error: 'datasetsCBR' folder not found inside 'data/'")
        print(f"Checked path: {data_dir}")
        return


    runner = ExperimentRunner(data_dir, output_dir)

    # Ask for confirmation before starting a long run
    print("\nThis will run a large number of experiments and may take several hours.")
    response = input("Do you want to proceed? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("[Runner] Execution cancelled by user.")
        return

    print("\n[Runner] Starting experiments...")

    # Run experiments with error handling
    try:
        fold_results, aggregated_results = runner.run_all_experiments()

        print("\n[Runner] All experiments completed successfully.")
        print(f"[Runner] Check '{output_dir.relative_to(PROJECT_ROOT)}' folder for output files.")

    except KeyboardInterrupt:
        print("\n\n[Runner] Experiments interrupted by user (Ctrl+C).")
        print(f"[Runner] Partial results (if any) saved in '{output_dir.relative_to(PROJECT_ROOT)}'.\n")

    except Exception as e:
        print(f"\n\n[Runner] An error occurred: {type(e).__name__}: {e}")
        print(f"[Runner] Check '{output_dir.relative_to(PROJECT_ROOT)}' folder for partial results.\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()