"""
This file provides three distance metrics for the k-IBL algorithm
as required by the assignment:
1. Euclidean
2. Cosine
3. HEOM (Heterogeneous Euclidean Overlap Metric)

All functions accept a `weights` vector for the fw_KIBLAlgorithm.
"""

import numpy as np


def euclidean_distance(x1: np.ndarray, x2: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculates the weighted Euclidean distance (L2 norm).
    d(q,x) = sqrt( sum( w_f * (q_f - x_f)^2 ) )

    This function is adapted to handle all kinds of attributes
    (numerical and categorical) because the parser.py script
    pre-processes all data into a numeric format (normalized [0,1] or label encoded).
    """
    squared_diff = (x1 - x2) ** 2
    weighted_squared_diff = weights * squared_diff
    sum_weighted = weighted_squared_diff.sum()
    distance = sum_weighted ** 0.5
    return distance


def cosine_distance(x1: np.ndarray, x2: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculates the weighted Cosine Distance (1 - Cosine Similarity).

    Similarity = (x1 . x2) / (||x1|| * ||x2||)

    Weights are applied before dot products and norm calculations.
    This function is also adapted for heterogeneous data as it expects
    all features to be numeric after pre-processing.
    """
    weighted_x1 = weights * x1
    weighted_x2 = weights * x2

    dot_product = (weighted_x1 * weighted_x2).sum()

    norm_x1 = (weighted_x1 ** 2).sum() ** 0.5
    norm_x2 = (weighted_x2 ** 2).sum() ** 0.5

    # Handle zero-vector case to avoid division by zero
    if norm_x1 == 0 or norm_x2 == 0:
        return 1.0  # Max distance if one vector is all zeros

    cosine_similarity = dot_product / (norm_x1 * norm_x2)

    # Clamp value to [-1.0, 1.0] to correct floating point inaccuracies
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

    cosine_distance = 1.0 - cosine_similarity
    return cosine_distance


def heom_distance(x1: np.ndarray, x2: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculates the Heterogeneous Euclidean Overlap Metric (HEOM).

    d(q,x) = sum( w_f * delta(q_f, x_f) )

    Justification: The assignment requires adapting this distance
    to handle both numerical and categorical data.
    Standard HEOM uses:
    - Overlap metric (0 if same, 1 if different) for categorical.
    - Normalized Euclidean distance for numerical.

    Since our pre-processing converts all features (numeric to [0, 1]
    and categorical to integer labels), we can use the L1 norm
    (Manhattan distance) as a justified adaptation. For normalized
    numeric features, |q_f - x_f| is the L1 distance. For label-encoded
    features, |q_f - x_f| is 0 if they are the same and >= 1 if they
    are different, which approximates the Overlap metric.

    This implementation: d(q,x) = sum( w_f * |q_f - x_f| )
    """
    absolute_diff = abs(x1 - x2)
    weighted_diff = weights * absolute_diff
    distance = weighted_diff.sum()
    return distance