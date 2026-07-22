"""MinGap — minimize the number of {1,2} blocks per row."""

from __future__ import annotations

import itertools

from mingap.score import block_cost
from nb1p import validate_ternary
from mingap._registry import available, get, register
from mingap.types import Matrix, OptimizeResult, Permutation

from mingap.analyze import analyze
__all__ = [
    "solve",
    "solve_all",
    "available",
    "register",
    "analyze",
    "block_cost",
    "Matrix",
    "OptimizeResult",
    "Permutation",
    "validate_ternary",
]



def solve(matrix: Matrix, *, solver: str = "auto") -> OptimizeResult:
    """Run the selected MinGap backend on *matrix*.

    Exact and heuristic backends return feasible solutions. Bound backends may
    instead return a certified objective bound and a placeholder permutation;
    see the selected backend's documentation.
    """
    return get(solver).solve(matrix)


def solve_all(matrix: Matrix, *, solver: str = "auto") -> list[Permutation]:
    """Enumerate every optimal MinGap permutation of *matrix*.

    The minimum cost is found with *solver*, then every column permutation
    achieving it is returned. Enumeration is :math:`O(n!)` over all permutations
    — no MinGap backend exposes an exact enumerator — so this is only
    practical for small matrices.

    Unlike :func:`solve`, the full matrix is passed to the solver (no row
    stripping) because the enumeration step must evaluate every permutation
    against the true cost anyway.
    """
    _, n = validate_ternary(matrix)
    if n <= 1:
        return [list(range(n))]

    best_cost = get(solver).solve(matrix).cost
    return [
        list(perm)
        for perm in itertools.permutations(range(n))
        if block_cost(matrix, list(perm)) == best_cost
    ]
