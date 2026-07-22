"""MinGap-specific types; matrix primitives re-exported from nb1p."""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

from nb1p.types import Matrix, Permutation

__all__ = [
    "Matrix",
    "Permutation",
    "OptimizeResult",
    "AnalysisReport",
    "MinGapSolver",
]


class OptimizeResult(NamedTuple):
    permutation: Permutation
    cost: int | float


class AnalysisReport(NamedTuple):
    is_nb1p: bool
    permutation: Permutation
    n_gaps: int


@runtime_checkable
class MinGapSolver(Protocol):
    def solve(self, matrix: Matrix) -> OptimizeResult: ...
