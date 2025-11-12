"""
This file contains the from-scratch implementations of three
Instance Reduction (IR) algorithms as required by the assignment:
1. Edited Nearest Neighbour (ENN)
2. Toussaint's Condensed Nearest Neighbour (TCNN)
3. Iterative Case Filtering (ICF)

These functions take a training set (X, y) and return a
new, smaller training set (X_reduced, y_reduced).
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple

# --- Type Aliases ---
DataSet = Tuple[np.ndarray, np.ndarray]


# -----------------------------------------------------------------
#  Algorithm 1: Edited Nearest Neighbour (ENN)
#  (From category: ENN, alIKNN, or MENN)
# -----------------------------------------------------------------

def enn(X: np.ndarray, y: np.ndarray, k: int = 3) -> DataSet:
    """
    Performs Edited Nearest Neighbour (ENN) reduction.

    ENN removes instances whose class label differs from the
    majority class of its k-nearest neighbors. It is primarily
    used for removing noise and smoothing decision boundaries.
    """
    print(f"  [IR] Running ENN (k={k})...")
    if k >= len(X):
        # k must be smaller than the number of samples
        print("  [IR] Warning: k is larger than dataset, returning original data.")
        return X, y

    # Use KNeighborsClassifier for an efficient neighbor search.
    # We use the default Euclidean distance (p=2).
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    # Get the predictions for the training set *itself*
    y_pred = knn.predict(X)

    # Create a boolean mask to keep only instances where
    # the true label matches the predicted label.
    mask = y == y_pred

    X_reduced = X[mask]
    y_reduced = y[mask]

    # Print statement for storage analysis
    reduction_pct = (1 - len(X_reduced) / len(X)) * 100
    print(f"    → ENN complete. Retained {len(X_reduced)}/{len(X)} ({reduction_pct:.1f}% reduction)")

    return X_reduced, y_reduced


# -----------------------------------------------------------------
#  Algorithm 2: Toussaint's Condensed Nearest (TCNN)
#  (From category: SNN, TCNN, or MCNN)
# -----------------------------------------------------------------

def _find_nearest_neighbor(instance: np.ndarray, X_set: np.ndarray) -> int:
    """
    Helper for TCNN: Finds the index of the nearest neighbor.

    This implementation uses Euclidean distance (L2 norm), as it is
    the standard for k-NN when not otherwise specified.
    """
    # Calculates Euclidean distance from 'instance' to all points in 'X_set'
    distances = np.linalg.norm(X_set - instance, axis=1)
    return np.argmin(distances)


def tcnn(X: np.ndarray, y: np.ndarray) -> DataSet:
    """
    Performs Toussaint's Condensed Nearest Neighbour (TCNN) reduction.

    TCNN is a fast, one-pass implementation of CNN. It finds a
    consistent subset (S) that can correctly classify the
    original set (T). It's good for removing redundant instances.
    """
    print("  [IR] Running TCNN...")

    # 1. Initialize S (the "store") with one instance of each class
    S_indices = []
    for c in np.unique(y):
        # Find the first occurrence of each class and add its index
        S_indices.append(np.where(y == c)[0][0])

    S = X[S_indices].tolist()  # Store S as a list for easy appending
    S_y = y[S_indices].tolist()

    # 2. Create N (the "not-in-S" set, or "grab bag")
    N_indices = [i for i in range(len(X)) if i not in S_indices]
    N = X[N_indices]
    N_y = y[N_indices]

    i = 0
    while i < len(N):
        # 3. Find nearest neighbor in S to the i-th instance in N
        x_n, y_n = N[i], N_y[i]

        # Convert S to numpy array for fast distance calculation
        S_np = np.array(S)
        nearest_idx_in_S = _find_nearest_neighbor(x_n, S_np)

        # 4. Check for misclassification
        if S_y[nearest_idx_in_S] != y_n:
            # Misclassified. Add (x_n, y_n) from N to S
            S.append(x_n)
            S_y.append(y_n)

            # Remove it from N
            N = np.delete(N, i, axis=0)
            N_y = np.delete(N_y, i, axis=0)

            # Reset the scan of N
            i = 0
        else:
            # Correctly classified. Move to next instance in N.
            i += 1

    X_reduced = np.array(S)
    y_reduced = np.array(S_y)

    # Print statement for storage analysis
    reduction_pct = (1 - len(X_reduced) / len(X)) * 100
    print(f"    → TCNN complete. Retained {len(X_reduced)}/{len(X)} ({reduction_pct:.1f}% reduction)")

    return X_reduced, y_reduced


# -----------------------------------------------------------------
#  Algorithm 3: Iterative Case Filtering (ICF)
#  (From category: IB3 or ICF)
# -----------------------------------------------------------------

def icf(X: np.ndarray, y: np.ndarray, k: int = 1) -> DataSet:
    """
    Performs Iterative Case Filtering (ICF) reduction.

    ICF identifies "good" instances. An instance is good if its
    k-NNs are from the same class (Coverage) and if it is part of
    the k-NNs of other "good" instances (Reachability).

    This implementation uses the common heuristic:
    Keep instance 'i' if Coverage(i) > Reachability(i)
    """
    print(f"  [IR] Running ICF (k={k})...")

    # Use k-NN for efficient neighbor search (Euclidean distance default)
    knn = KNeighborsClassifier(n_neighbors=k + 1)  # +1 to include self
    knn.fit(X, y)

    # Get (k+1) neighbors for all instances
    distances, indices = knn.kneighbors(X)

    # Remove self from neighbor list
    neighbors_idx = indices[:, 1:]  # Shape (n_samples, k)

    # Calculate Coverage and Reachability
    coverage = np.zeros(len(X), dtype=int)
    reachability = np.zeros(len(X), dtype=int)

    for i in range(len(X)):
        # Get labels of the k-NNs
        neighbor_labels = y[neighbors_idx[i]]

        # --- Coverage ---
        # How many of i's k-NNs have the same class as i?
        coverage[i] = np.sum(neighbor_labels == y[i])

        # --- Reachability ---
        # Who is instance 'i' a neighbor to?
        # Find all instances 'j' where 'i' is in neighbors_idx[j]
        reachable_by_indices = np.where(neighbors_idx == i)[0]

        if len(reachable_by_indices) > 0:
            # Get labels of instances that can "reach" i
            reachable_by_labels = y[reachable_by_indices]
            # How many of *them* have the same class as i?
            reachability[i] = np.sum(reachable_by_labels == y[i])

    # Heuristic: An instance 'i' is kept if its Coverage > Reachability.
    # Intuition: We keep instances that are "good classifiers" (high Coverage)
    # but are not "core instances" that many others rely on (low Reachability),
    # thus removing redundancy.
    mask = coverage > reachability

    X_reduced = X[mask]
    y_reduced = y[mask]

    # Print statement for storage analysis
    reduction_pct = (1 - len(X_reduced) / len(X)) * 100
    print(f"    → ICF complete. Retained {len(X_reduced)}/{len(X)} ({reduction_pct:.1f}% reduction)")

    return X_reduced, y_reduced