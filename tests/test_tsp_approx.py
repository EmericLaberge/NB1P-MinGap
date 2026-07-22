"""Upper-bound checks for the tsp_approx (Christofides/CBM) heuristic."""

from __future__ import annotations

import pytest

import mingap
from test_mingap import CROSS_CHECK_MATRICES, _random_matrices


@pytest.fixture(scope="module")
def tsp_approx_solver():
    if "tsp_approx" not in mingap.available():
        pytest.skip("MinGap backend 'tsp_approx' requires the tsp extra")
    return mingap.get("tsp_approx")


@pytest.mark.parametrize("matrix", CROSS_CHECK_MATRICES + _random_matrices())
def test_tsp_approx_is_upper_bound(tsp_approx_solver, matrix):
    optimum = mingap.get("brute_force").solve(matrix).cost
    result = tsp_approx_solver.solve(matrix)

    # Feasible permutation → its cost is an upper bound on the optimum.
    # NOTE: this includes cost-0 (satisfiable) matrices — tsp_approx is a
    # 1.5-approximation for the CBM objective, not for MinGap, so it may
    # return > 0 on a satisfiable instance; only the UB invariant holds.
    assert result.cost >= optimum
    n = len(matrix[0]) if matrix else 0
    assert result.cost <= n  # trivial per-column bound


@pytest.mark.parametrize("matrix", CROSS_CHECK_MATRICES + _random_matrices())
def test_tsp_approx_returns_valid_permutation(tsp_approx_solver, matrix):
    n = len(matrix[0]) if matrix else 0
    result = tsp_approx_solver.solve(matrix)

    assert sorted(result.permutation) == list(range(n))


def test_tsp_approx_is_deterministic(tsp_approx_solver):
    for matrix in CROSS_CHECK_MATRICES:
        first = tsp_approx_solver.solve(matrix)
        second = tsp_approx_solver.solve(matrix)
        assert first == second
