"""Tests for MinGap optimisation across all installed backends."""

from __future__ import annotations

import random

import pytest

import nb1p
import mingap
from mingap import analyze
from mingap.score import block_cost

# Hand-picked matrices, including ones whose optimum forces multiple blocks
# and unsatisfiable gadgets.
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


def test_zero_cost_on_nb1p_matrix(mingap_solver, paper_matrix):
    if mingap_solver == "greedy":
        pytest.skip("paper_matrix is non-trivial; greedy is a heuristic and may exceed the optimum")
    result = mingap.solve(paper_matrix, solver=mingap_solver)
    assert result.cost == 0
    assert nb1p.verify(paper_matrix, result.permutation)


def test_result_unpacks(mingap_solver, satisfiable_matrix):
    perm, cost = mingap.solve(satisfiable_matrix, solver=mingap_solver)
    assert cost == 0
    assert block_cost(satisfiable_matrix, perm) == 0


def test_optimum_matches_brute_force(mingap_solver, unsatisfiable_matrix):
    expected = mingap.solve(
        unsatisfiable_matrix, solver="brute_force").cost
    got = mingap.solve(unsatisfiable_matrix, solver=mingap_solver)
    assert got.cost == expected
    assert block_cost(unsatisfiable_matrix, got.permutation) == got.cost


def test_unsatisfiable_has_positive_cost(unsatisfiable_matrix):
    result = mingap.solve(unsatisfiable_matrix, solver="brute_force")
    assert result.cost >= 1


def test_cost_zero_iff_nb1p(unsatisfiable_matrix, satisfiable_matrix):
    assert (mingap.solve(satisfiable_matrix, solver="brute_force").cost == 0) \
        == nb1p.check(satisfiable_matrix)
    assert (mingap.solve(unsatisfiable_matrix, solver="brute_force").cost == 0) \
        == nb1p.check(unsatisfiable_matrix)


def test_solve_all_returns_optimal_perms(satisfiable_matrix):
    sols = mingap.solve_all(satisfiable_matrix, solver="brute_force")
    assert len(sols) >= 1
    for perm in sols:
        assert block_cost(satisfiable_matrix, perm) == 0


@pytest.mark.parametrize("matrix", CROSS_CHECK_MATRICES + _random_matrices())
def test_backend_matches_brute_force_optimum(mingap_solver, matrix):
    if mingap_solver == "greedy":
        pytest.skip("greedy is a heuristic; lower-bound invariant covered by test_greedy_meets_lower_bound")
    reference = mingap.solve(matrix, solver="brute_force").cost
    result = mingap.solve(matrix, solver=mingap_solver)
    assert result.cost == reference
    assert block_cost(matrix, result.permutation) == result.cost


@pytest.mark.parametrize("matrix", CROSS_CHECK_MATRICES + _random_matrices())
def test_greedy_meets_lower_bound(mingap_solver, matrix):
    if mingap_solver != "greedy":
        pytest.skip("lower-bound invariant only applies to heuristic backends")
    reference = mingap.solve(matrix, solver="brute_force").cost
    result = mingap.solve(matrix, solver=mingap_solver)
    assert result.cost >= reference, (
        f"greedy ({result.cost}) beat brute force ({reference}) on {matrix}"
    )
    assert block_cost(matrix, result.permutation) == result.cost


def test_block_cost_with_2_glue():
    """A matrix where 2-as-glue makes cost 0."""
    M = [
        [1, 2, 0, 1, 0],
        [0, 1, 2, 0, 1],
    ]
    result = mingap.solve(M, solver="brute_force")
    assert result.cost == 0


def test_analyze_nb1p(paper_matrix):
    report = analyze(paper_matrix)
    assert report.is_nb1p
    assert report.n_gaps == 0
    assert nb1p.verify(paper_matrix, report.permutation)


def test_analyze_non_nb1p(unsatisfiable_matrix):
    report = analyze(unsatisfiable_matrix)
    assert not report.is_nb1p
    assert report.n_gaps >= 1
    assert block_cost(unsatisfiable_matrix, report.permutation) == report.n_gaps

def test_analyze_report_unpacks(satisfiable_matrix):
    is_nb1p, _, n_gaps = analyze(satisfiable_matrix)
    assert is_nb1p and n_gaps == 0


def test_maxsat_fuzz_1000_n_lt_6():
    """MaxSAT (v1, paper version) must match brute force on 1000 random matrices with n<6."""
    from mingap import _registry as mingap_registry

    if "maxsat" not in mingap_registry.available():
        pytest.skip("MaxSAT backend not installed (need `pip install mingap[sat]`)")

    rng = random.Random(20260721)
    n_matrices = 1000
    mismatches: list[tuple[int, list[list[int]], int, int]] = []
    progress_every = 100

    for i in range(n_matrices):
        n = rng.randint(2, 5)  # n < 6
        m = rng.randint(1, max(1, 2 * n))
        matrix = nb1p.random_matrix(m, n, p1=0.3, p0=0.5, rng=rng)

        bf = mingap.solve(matrix, solver="brute_force")
        ms = mingap.solve(matrix, solver="maxsat")

        if (i + 1) % progress_every == 0:
            print(f"\n[fuzz] {i + 1}/{n_matrices} processed, {len(mismatches)} mismatches so far", flush=True)

        if ms.cost != bf.cost:
            mismatches.append((i, matrix, bf.cost, ms.cost))

    print(f"\n[fuzz] done: {n_matrices} matrices, {len(mismatches)} mismatches", flush=True)
    assert not mismatches, (
        f"MaxSAT disagreed with brute force on {len(mismatches)}/{n_matrices} matrices; "
        f"first mismatch: idx={mismatches[0][0]} n={len(mismatches[0][1][0])} "
        f"bf={mismatches[0][2]} ms={mismatches[0][3]}"
    )
