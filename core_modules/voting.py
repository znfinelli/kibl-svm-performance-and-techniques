"""
This file implements the two voting policies required by the assignment:
1. Modified Plurality
2. Borda Count
"""

import numpy as np
from typing import List, Any, Tuple, Dict

# Type Alias for clarity
Neighbor = Tuple[float, int, Any]  # (distance, index, label)


def modified_plurality_vote(k_neighbors: List[Neighbor]) -> Any:
    """
    Calculates the winner using Modified Plurality.

    Computes a standard plurality vote. In case of a tie, it removes
    the farthest neighbor (last in the list) and re-votes. This
    process repeats until a single winner is found or only one
    neighbor remains.
    """
    k = len(k_neighbors)
    if k == 0:
        return 0 # Or raise error

    # Start with all k neighbors
    current_neighbors = k_neighbors.copy()

    # The k_neighbors list is assumed to be sorted by similarity (closest first).
    # In case of a tie, we remove the farthest neighbor (last in the list)
    # and re-compute the vote.
    while len(current_neighbors) > 0:
        # Count votes for the current set of neighbors
        vote_counts: Dict[Any, int] = {}
        for distance, idx, label in current_neighbors:
            vote_counts[label] = vote_counts.get(label, 0) + 1

        if not vote_counts:
            # Should not happen, but as a fallback, return closest
            return k_neighbors[0][2]

        # Find the winner(s)
        max_votes = max(vote_counts.values())
        winners = [label for label, count in vote_counts.items() if count == max_votes]

        # Case 1: Clear winner
        if len(winners) == 1:
            return winners[0]

        # Case 2: Tie, and only one neighbor left. Return its class.
        if len(current_neighbors) == 1:
            return current_neighbors[0][2]

        # Case 3: Tie, multiple neighbors left. Remove the farthest and repeat.
        current_neighbors = current_neighbors[:-1]

    # Fallback (e.g., if k=0 or list becomes empty)
    # Return the class of the closest neighbor.
    return k_neighbors[0][2]


def borda_count_vote(k_neighbors: List[Neighbor]) -> Any:
    """
    Calculates the winner using Borda Count.

    Assigns points based on rank. The most similar instance (rank 0)
    gets k-1 points, the next gets k-2, ..., down to the k-th
    instance which gets 0 points (k-k).

    The winner is the class with the highest total points.
    """
    k = len(k_neighbors)
    if k == 0:
        return None  # Or raise error

    points: Dict[Any, int] = {}

    # Assign points based on rank
    for rank, (distance, idx, label) in enumerate(k_neighbors):
        # The most similar instance (rank 0) gets k-1 points.
        score = k - (rank + 1)  # (rank 0 gets k-1, rank 1 gets k-2, ...)
        points[label] = points.get(label, 0) + score

    # Find the winner(s)
    max_points = max(points.values())
    winners = [label for label, pts in points.items() if pts == max_points]

    # Case 1: Clear winner
    if len(winners) == 1:
        return winners[0]

    # Case 2: Tie.
    # Justification: The assignment requires a method for breaking ties.
    # This implementation selects the class of the *most similar* neighbor
    # (rank 0), which is a simple and deterministic tie-breaking rule.
    else:
        return k_neighbors[0][2]