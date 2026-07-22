"""Type aliases and solver protocols for the library.
Semantic names for the plain structures we pass around, plus the
duck-typed interface every solver backend needs to implement.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

#: A ternary matrix ``M ∈ {0,1,2}^{m×n}`` as a list of rows.
#: ``0`` = present, ``1`` = lost here, ``2`` = ancestral loss (wildcard).
Matrix = list[list[int]]

#: A column permutation ``π`` where ``perm[k]`` = column index at position ``k``.
Permutation = list[int]


class SolveResult(NamedTuple):
    """Outcome of an NB1P decision query."""

    satisfiable: bool
    permutation: Permutation | None






@runtime_checkable
class NB1PSolver(Protocol):
    """Interface for an NB1P (decision) backend."""

    def solve(self, matrix: Matrix) -> SolveResult: ...

    def solve_all(self, matrix: Matrix) -> list[Permutation]: ...


