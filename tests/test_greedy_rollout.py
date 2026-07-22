"""Tests for the rollout greedy MinGap solver."""

from __future__ import annotations

import pytest

import mingap
from mingap._greedy_rollout import GreedyRolloutSolver
from mingap.score import block_cost
from test_greedy import CROSS_CHECK_MATRICES, _random_matrices

# The solver's ProcessPoolExecutor forks from pytest's multi-threaded parent
# (numba threads) — Python 3.13 warns per fork. Children do pure CPU work on
# inherited data, no locks cross the fork: noise, not a hazard.
pytestmark = pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)


@pytest.fixture(scope="module")
def rollout():
    # Small pool per solve keeps the parametrized suite fast.
    return GreedyRolloutSolver(k=6, n_jobs=2)


def _all_matrices():
    return CROSS_CHECK_MATRICES + _random_matrices()


# ---------- Registration ----------


def test_greedy_rollout_is_registered():
    assert "greedy_rollout" in mingap.available()
    assert mingap.get("greedy_rollout").__class__ is GreedyRolloutSolver


# ---------- Edge cases ----------


def test_empty_matrix():
    out = GreedyRolloutSolver().solve([])
    assert out == ([], 0)


def test_no_rows():
    out = GreedyRolloutSolver().solve([[0, 0, 0]])
    assert out.cost == 0
    assert sorted(out.permutation) == [0, 1, 2]


def test_single_column():
    out = GreedyRolloutSolver().solve([[1], [0], [2]])
    assert out == ([0], 0)


def test_single_worker():
    """n_jobs=1 takes the in-process path and still works."""
    solver = GreedyRolloutSolver(k=4, n_jobs=1)
    matrix = CROSS_CHECK_MATRICES[0]
    out = solver.solve(matrix)
    assert out.cost >= mingap.get("brute_force").solve(matrix).cost


# ---------- Invariants vs greedy and brute force ----------


@pytest.mark.parametrize("matrix", _all_matrices())
def test_rollout_at_least_as_good_as_greedy(rollout, matrix):
    """Strategy 0 reproduces the plain greedy baseline, so the rollout
    minimum can never be worse than a single greedy run."""
    assert rollout.solve(matrix).cost <= mingap.get("greedy").solve(matrix).cost


@pytest.mark.parametrize("matrix", _all_matrices())
def test_rollout_is_upper_bound(rollout, matrix):
    """Rollout is still a heuristic: never better than brute force."""
    optimum = mingap.get("brute_force").solve(matrix).cost
    assert rollout.solve(matrix).cost >= optimum


@pytest.mark.parametrize("matrix", _all_matrices())
def test_rollout_returns_valid_permutation(rollout, matrix):
    n = len(matrix[0]) if matrix else 0
    out = rollout.solve(matrix)
    assert sorted(out.permutation) == list(range(n))


@pytest.mark.parametrize("matrix", _all_matrices())
def test_rollout_cost_self_consistent(rollout, matrix):
    out = rollout.solve(matrix)
    assert block_cost(matrix, out.permutation) == out.cost


def test_rollout_is_deterministic(rollout):
    for matrix in CROSS_CHECK_MATRICES:
        assert rollout.solve(matrix) == rollout.solve(matrix)
