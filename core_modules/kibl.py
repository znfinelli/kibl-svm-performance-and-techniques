"""
This file contains the from-scratch implementation of the k-Instance-Based
Learning (k-IBL) algorithm as required by the assignment.
It is designed to be modular, accepting distance, voting, and retention
functions/policies as parameters.
"""

import numpy as np
import time
from typing import Callable, List, Tuple, Any, Optional, Dict

# --- Type Aliases ---
Instance = Tuple[np.ndarray, Any]  # (features, label)
Neighbor = Tuple[float, int, Any]  # (distance, index, label)


class kIBLAlgorithm:
    """
    Implements a k-Instance-Based Learning (k-IBL) classifier.

    This class stores a training instance base and classifies new query
    instances by finding the k-nearest neighbors and applying a
    voting policy. It also supports various retention policies
    for adding classified instances back into the instance base.
    """

    def __init__(self,
                 k: int,
                 distance_func: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
                 voting_func: Callable[[List[Neighbor]], Any],
                 retention_policy: str = 'NR'):
        """
        Initializes the k-IBL algorithm.

        Args:
            k (int): The number of neighbors to retrieve (e.g., 3, 5, 7).
            distance_func (Callable): The function to use for distance calculation
                                     (e.g., euclidean_distance).
            voting_func (Callable): The function to use for voting
                                  (e.g., modified_plurality_vote).
            retention_policy (str): The policy for retaining new instances
                                    ('NR', 'AR', 'DC', 'DD').
        """
        self.k: int = k
        self.distance_func: Callable = distance_func
        self.voting_func: Callable = voting_func
        self.retention_policy: str = retention_policy
        self.instance_base: List[Instance] = []

        # Feature weights.
        # Default weights are 1.0 (equal weighting) as required for the initial k-IBL analysis (Step 3).
        # These weights are modified later in fw_KIBLAlgorithm (Step 5).
        self.weights: Optional[np.ndarray] = None

        # This is needed to calculate the Storage Metric (Step 6.d) accurately.
        self.original_train_size: int = 0


    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Loads the training data into the instance base.
        This is the "learning" part of lazy learning.
        """
        n_features = X_train.shape[1]

        # Default weights are 1 for all features
        if self.weights is None:
            self.weights = np.ones(n_features)

        self.instance_base = [(X_train[i], y_train[i]) for i in range(len(X_train))]
        # Store size for storage metric
        self.original_train_size = len(X_train)

    def find_k_nearest(self, query_instance: np.ndarray) -> List[Neighbor]:
        """
        Finds the K nearest neighbors from the instance base for a query.

        The resulting list is ordered by distance (most similar first).
        """
        distances: List[Neighbor] = []

        for idx, (instance_features, instance_label) in enumerate(self.instance_base):
            # Calculate distance using the provided weighted distance function
            dist = self.distance_func(query_instance, instance_features, self.weights)
            distances.append((dist, idx, instance_label))

        # Sort by distance (the first element of the tuple)
        distances.sort(key=lambda x: x[0])

        # Return the top k neighbors
        k_neighbors = distances[:self.k]
        return k_neighbors

    def predict(self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Classifies all instances in a test set and measures performance.

        Args:
            X_test (np.ndarray): The test features (TestMatrix).
            y_test (Optional[np.ndarray]): The true test labels. Required for
                                           retention policies.

        Returns:
            Tuple[np.ndarray, float]:
                - An array of predictions.
                - The average problem-solving time per instance (efficiency metric).
        """
        predictions: List[Any] = []
        start_time = time.time()

        # Print statement to track execution progress
        print(f"  [K-IBL] Starting prediction with K={self.k}, Policy='{self.retention_policy}'. "
              f"Initial IB size: {len(self.instance_base)} instances.")

        for i in range(len(X_test)):
            query = X_test[i]
            # True label is needed for retention policies
            true_label = y_test[i] if y_test is not None else None

            # 1. Find k-nearest neighbors
            k_neighbors = self.find_k_nearest(query)

            # 2. Get prediction from voting policy
            pred = self.voting_func(k_neighbors)
            predictions.append(pred)

            # 3. Apply retention policy (if not 'NR' and true labels are known)
            if self.retention_policy != 'NR' and true_label is not None:
                self._apply_retention(query, pred, true_label, k_neighbors)

        # Calculate efficiency metric
        elapsed = time.time() - start_time
        time_per_instance = elapsed / len(X_test) if len(X_test) > 0 else 0

        # Print statement for efficiency and storage analysis
        final_size = len(self.instance_base)
        print(f"  [K-IBL] Prediction complete. Final IB size: {final_size} instances. "
              f"Total time: {elapsed:.3f}s. Avg time/instance: {time_per_instance:.6f}s.")

        return np.array(predictions), time_per_instance

    def _apply_retention(self,
                         query: np.ndarray,
                         predicted_label: Any,
                         true_label: Any,
                         k_neighbors: Optional[List[Neighbor]] = None) -> None:
        """
        Applies the chosen retention policy to the instance base.
        """
        # Never Retain (NR): The algorithm never retains the current instance q.
        # This is handled by default (do nothing).

        # Always Retain (AR): The algorithm retains all new solved instances.
        if self.retention_policy == 'AR':
            self.instance_base.append((query, true_label))

        # Different Class (DC): Retain instances that were solved incorrectly.
        elif self.retention_policy == 'DC':
            if predicted_label != true_label:
                self.instance_base.append((query, true_label))

        # Degree of Disagreement (DD): Retain based on neighbor disagreement.
        elif self.retention_policy == 'DD':
            if k_neighbors is None:
                return

            disagreement = self._calculate_disagreement(k_neighbors)

            # Implementation Decision: Threshold for retention.
            # If the disagreement 'd' is greater than this threshold, the instance is retained.
            # This is a justifiable implementation detail as the paper does not specify a threshold.
            threshold = 0.3

            if disagreement > threshold:
                self.instance_base.append((query, true_label))

            # Optional: Uncomment for detailed debugging of the DD policy
            # print(f"    [DD Policy] Disagreement: {disagreement:.4f}. Retained: {disagreement > threshold}")


    def _calculate_disagreement(self, k_neighbors: List[Neighbor]) -> float:
        """
        Calculates the Degree of Disagreement (d) based on the formula.
        d = #remaining_cases / ((#classes - 1) * #majority_cases)
        """

        # Count votes to find #majority_cases and #remaining_cases
        vote_counts: Dict[Any, int] = {}
        for distance, idx, label in k_neighbors:
            vote_counts[label] = vote_counts.get(label, 0) + 1

        if not vote_counts:
            return 0.0

        # #majority_cases is the number of instances with the most assigned class in K.
        majority_cases = max(vote_counts.values())

        total_neighbors = len(k_neighbors)
        # #remaining_cases is the number of instances with classes other than the majority one.
        remaining_cases = total_neighbors - majority_cases

        # Get total number of classes in the *entire instance base*
        # (The assignment specifies using all classes in the instance base).
        all_labels = set(label for features, label in self.instance_base)
        num_classes = len(all_labels) # #classes is the number of different classes in the instance base.

        # Handle edge cases
        if num_classes <= 1:
            return 0.0 # No disagreement possible with one class

        if majority_cases == 0:
            return 1.0 # Disagreement d reaches 1.0 when instances disagree the most.

        # #classes - 1 is introduced to normalize disagreement from 0 to 1.
        denominator = (num_classes - 1) * majority_cases

        if denominator == 0:
            return 0.0 # Avoid division by zero

        disagreement = remaining_cases / denominator

        # Normalize to [0, 1] range as per description
        disagreement = min(disagreement, 1.0)

        return disagreement

    def get_storage_percentage(self) -> float:
        """
        Calculates the current instance base size as a percentage
        of the original training set size.
        This is one of the required performance metrics.
        """
        # Use the stored original size
        if self.original_train_size == 0:
            return 0.0

        current_size = len(self.instance_base)
        # The storage of the kIBLAlgorithm (without reduction) is 100%.
        # This metric is most useful for retention policies and IR.
        return (current_size / self.original_train_size) * 100.0