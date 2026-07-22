"""Matrix in → NB1P check or minimum-block-cost permutation (one MinGap solve)."""

from __future__ import annotations

from mingap.types import AnalysisReport, Matrix


def analyze(matrix: Matrix, *, solver: str = "auto") -> AnalysisReport:
    """NB1P when cost == 0, else a minimum-block-cost permutation.

    A cost of 0 means every row forms a single loss block (all rows are
    segmental), which is exactly the NB1P property.

    ``mingap`` depends on :func:`mingap.score.block_cost` for scoring.
    """
    from mingap import solve as _solve

    result = _solve(matrix, solver=solver)
    return AnalysisReport(
        is_nb1p=(result.cost == 0),
        permutation=result.permutation,
        n_gaps=result.cost,
    )
