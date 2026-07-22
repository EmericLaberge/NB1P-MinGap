"""Tests for the greedy heuristic MinGap solver."""

from __future__ import annotations

import random
import time

import pytest

import mingap
import nb1p
from mingap.score import block_cost
from mingap._greedy import GreedySolver


# Mirrored from tests/test_mingap.py to keep tests/ package-free.
CROSS_CHECK_MATRICES = [
    [[1, 0, 1], [0, 1, 1], [1, 1, 0]],          # 3-cycle
    [[1, 0, 0, 1]],                              # forced blocks
    [[1, 0, 1, 0, 1]],
    [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]],
    nb1p.paper_example(),
]


def _random_matrices(n_each: int = 6):
    rng = random.Random(7)
    out = []
    for m in range(2, 7):
        for n in range(m, 7):
            for _ in range(n_each):
                out.append(nb1p.random_matrix(m, n, p1=0.3, p0=0.5, rng=rng))
    return out


def _all_matrices():
    return CROSS_CHECK_MATRICES + _random_matrices()


# ---------- Edge cases ----------


def test_empty_matrix():
    """n == 0 → empty permutation, cost 0."""
    out = GreedySolver().solve([])
    assert out == ([], 0)


def test_no_rows():
    """A zero-only matrix (every entry 0) yields cost 0 for any perm."""
    mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    out = GreedySolver().solve(mat)
    assert out.cost == 0
    assert sorted(out.permutation) == [0, 1, 2]


def test_single_column():
    """n == 1 → trivial."""
    out = GreedySolver().solve([[1], [0], [2]])
    assert out == ([0], 0)


# ---------- Cross-check vs brute force (lower-bound invariant) ----------


@pytest.mark.parametrize("matrix", _all_matrices())
def test_greedy_ge_brute_force(matrix):
    """Greedy is a heuristic: never better than brute force."""
    bf = mingap.solve(matrix, solver="brute_force").cost
    gr = mingap.solve(matrix, solver="greedy").cost
    assert gr >= bf, (
        f"greedy ({gr}) beat brute force ({bf}) on {matrix}"
    )


@pytest.mark.parametrize("matrix", _all_matrices())
def test_greedy_cost_self_consistent(matrix):
    """Reported cost must equal re-computed block_cost on the returned perm."""
    out = GreedySolver().solve(matrix)
    assert block_cost(matrix, out.permutation) == out.cost


@pytest.mark.parametrize("matrix", _all_matrices())
def test_greedy_returns_valid_permutation(matrix):
    """Permutation must be a full rearrangement of range(n)."""
    n = len(matrix[0])
    out = GreedySolver().solve(matrix)
    assert sorted(out.permutation) == list(range(n))


# ---------- Optimality on satisfiable matrices ----------

def test_greedy_zero_on_2_glue():
    """A trivially glue-soluble matrix: 2s bridge all 1s."""
    M = [
        [1, 2, 1],
        [2, 1, 2],
        [1, 2, 1],
    ]
    bf = mingap.solve(M, solver="brute_force").cost
    out = GreedySolver().solve(M)
    assert bf == 0  # sanity: the optimum is zero
    assert out.cost == 0


def test_greedy_on_paper_example():
    """The NB1P paper example: greedy must not beat the brute-force optimum."""
    mat = nb1p.paper_example()
    bf = mingap.solve(mat, solver="brute_force").cost
    out = GreedySolver().solve(mat)
    # Greedy is a heuristic — only the lower-bound invariant is required.
    # (Achieving zero here would be a bonus; the paper example has many
    # non-trivial permutations and best-fit insertion is not always enough.)
    assert out.cost >= bf


def test_greedy_zero_on_satisfiable_fixture():
    """A small satisfiable matrix has cost 0 optimum."""
    M = [
        [1, 1, 0],
        [0, 1, 1],
    ]
    out = GreedySolver().solve(M)
    assert out.cost == 0


# ---------- Performance ----------


def test_greedy_fast_on_large_matrix():
    """50 columns × 30 rows must finish in well under 1 second (warm)."""
    rng = random.Random(11)
    mat = nb1p.random_matrix(30, 50, p1=0.3, p0=0.5, rng=rng)
    # Warm numba JIT and one algorithm pass to remove cold start from timing.
    GreedySolver().solve(mat)
    t0 = time.perf_counter()
    out = GreedySolver().solve(mat)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"greedy took {elapsed:.2f}s on 30x50"
    # Sanity: cost is finite and perm is a valid rearrangement.
    n = len(mat[0])
    assert sorted(out.permutation) == list(range(n))
    assert out.cost >= 0
