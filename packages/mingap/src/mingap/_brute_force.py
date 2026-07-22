"""Brute-force MinGap — try all n! permutations, pick the cheapest.
Reference implementation. Usable up to n ≈ 10 (10! = 3.6M). No extra deps.
Uses numba when ``mingap[sa]`` is installed.
"""

from __future__ import annotations

import itertools

from mingap.score import block_cost
from nb1p import validate_ternary
from mingap._registry import register
from mingap.types import Matrix, OptimizeResult

try:
    from mingap._fast import brute_force_best, matrix_to_array

    _HAS_FAST = True
except ImportError:
    _HAS_FAST = False


@register("brute_force")
class BruteForceSolver:
    """Exhaustive inversion-aware minimum-gap search."""

    def solve(self, matrix: Matrix) -> OptimizeResult:
        _, n = validate_ternary(matrix)
        if n <= 1:
            return OptimizeResult(list(range(n)), 0)

        if _HAS_FAST:
            m = len(matrix)
            A = matrix_to_array(matrix)
            best_perm, best_score = brute_force_best(A, m, n)
            return OptimizeResult(best_perm.tolist(), int(best_score))

        best_score = len(matrix) * n + 1
        best_perm = list(range(n))
        for perm in itertools.permutations(range(n)):
            s = block_cost(matrix, list(perm))
            if s < best_score:
                best_score = s
                best_perm = list(perm)
                if s == 0:
                    break
        return OptimizeResult(best_perm, best_score)
