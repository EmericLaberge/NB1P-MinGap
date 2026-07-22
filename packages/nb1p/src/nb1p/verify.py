"""O(mn) NB1P certificate check.
Given M ∈ {0,1,2}^{m×n} and a permutation π of [n], verify that in
every row the span between the leftmost and rightmost 1 is zero-free
(2s are wildcards).
"""

from __future__ import annotations

from nb1p.types import Matrix, Permutation


def verify(matrix: Matrix, perm: Permutation) -> bool:
    """Check *perm* satisfies NB1P on *matrix*.
    Args:
        matrix: m×n ternary matrix (list of rows of 0/1/2).
        perm:   permutation of [0..n-1]; perm[k] = column at position k.
    Returns:
        True iff π is a valid NB1P permutation.
    """
    n = len(perm)
    for row in matrix:
        leftmost = None
        rightmost = None
        for k in range(n):
            if row[perm[k]] == 1:
                if leftmost is None:
                    leftmost = k
                rightmost = k

        if leftmost is None or leftmost == rightmost:
            continue

        for k in range(leftmost, rightmost + 1):
            if row[perm[k]] == 0:
                return False

    return True
