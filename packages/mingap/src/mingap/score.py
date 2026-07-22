"""Block cost: segmental-block gap scoring for ternary matrices.

Computes the number of extra segmental blocks beyond the first per row.
An active block is a maximal contiguous run of {1,2} in the permuted row
that contains at least one real 1. Value 2 acts as transparent glue —
a lone 2-only block between two 0-regions does not create an additional
segment, because the absent gene can be assigned to either adjacent block.

Cost per row = max(0, active_blocks - 1).
This equals the minimum number of additional segmental deletion events
beyond the first, which is the MinGap optimization objective.
"""

from __future__ import annotations

from mingap.types import Matrix, Permutation

try:
    from mingap._fast import block_cost_njit, matrix_to_array, permutation_to_array

    _HAS_FAST = True
except ImportError:
    _HAS_FAST = False


def block_cost(matrix: Matrix, perm: Permutation) -> int:
    """Block cost: sum over rows of max(0, active_blocks - 1).
    An active block is a maximal contiguous run of {1, 2} in the
    permuted row that contains at least one 1. Value 2 acts as glue
    between adjacent 1s — a lone 2-only block contributes nothing.
    O(mn).  Uses numba when ``mingap[sa]`` is installed.
    """
    if _HAS_FAST:
        m = len(matrix)
        if m == 0:
            return 0
        n = len(perm)
        A = matrix_to_array(matrix)
        p = permutation_to_array(perm)
        return int(block_cost_njit(A, p, m, n))

    n = len(perm)
    total = 0
    for row in matrix:
        blocks_with_one = 0
        in_block = False
        has_one = False
        for k in range(n):
            v = row[perm[k]]
            if v == 0:
                if in_block and has_one:
                    blocks_with_one += 1
                in_block = False
                has_one = False
            else:
                in_block = True
                if v == 1:
                    has_one = True
        if in_block and has_one:
            blocks_with_one += 1
        total += max(0, blocks_with_one - 1)
    return total
