"""Brute-force NB1P — try all n! permutations, check each in O(mn).
Reference implementation. Usable up to n ≈ 10 (11! = 39M). No extra deps.
"""

from __future__ import annotations

import itertools

from nb1p import validate_ternary, verify
from nb1p._registry import register
from nb1p.types import Matrix, Permutation, SolveResult


@register("brute_force")
class BruteForceSolver:
    """Exhaustive permutation search."""

    def solve(self, matrix: Matrix) -> SolveResult:
        _, n = validate_ternary(matrix)
        if n <= 1:
            return SolveResult(True, list(range(n)))
        for perm in itertools.permutations(range(n)):
            if verify(matrix, list(perm)):
                return SolveResult(True, list(perm))
        return SolveResult(False, None)

    def solve_all(self, matrix: Matrix) -> list[Permutation]:
        _, n = validate_ternary(matrix)
        if n <= 1:
            return [list(range(n))]
        return [
            list(perm)
            for perm in itertools.permutations(range(n))
            if verify(matrix, list(perm))
        ]
