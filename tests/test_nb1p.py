"""Tests for NB1P decision and enumeration across all installed backends."""

from __future__ import annotations

import nb1p


def test_check_is_dependency_free(paper_matrix):
    assert nb1p.check(paper_matrix) is True


def test_check_detects_unsatisfiable(unsatisfiable_matrix):
    assert nb1p.check(unsatisfiable_matrix) is False


def test_solve_satisfiable(nb1p_solver, paper_matrix):
    result = nb1p.solve(paper_matrix, solver=nb1p_solver)
    assert result.satisfiable
    assert nb1p.verify(paper_matrix, result.permutation)


def test_solve_unsatisfiable(nb1p_solver, unsatisfiable_matrix):
    result = nb1p.solve(unsatisfiable_matrix, solver=nb1p_solver)
    assert not result.satisfiable
    assert result.permutation is None


def test_solve_result_unpacks(nb1p_solver, satisfiable_matrix):
    ok, perm = nb1p.solve(satisfiable_matrix, solver=nb1p_solver)
    assert ok
    assert nb1p.verify(satisfiable_matrix, perm)


def test_solve_all_matches_brute_force(nb1p_solver, paper_matrix):
    expected = {
        tuple(p)
        for p in nb1p.solve_all(paper_matrix, solver="brute_force")
    }
    got = {
        tuple(p)
        for p in nb1p.solve_all(paper_matrix, solver=nb1p_solver)
    }
    assert got == expected


def test_paper_example_has_twelve_solutions(paper_matrix):
    perms = nb1p.solve_all(paper_matrix, solver="brute_force")
    assert len(perms) == 12


def test_auto_selects_a_backend(satisfiable_matrix):
    result = nb1p.solve(satisfiable_matrix, solver="auto")
    assert result.satisfiable


def test_unknown_solver_raises(satisfiable_matrix):
    import pytest

    with pytest.raises(ValueError):
        nb1p.solve(satisfiable_matrix, solver="does_not_exist")


def test_custom_backend_registration():
    from nb1p._registry import register, get
    from nb1p.types import SolveResult

    @register("dummy_identity")
    class _Dummy:
        def solve(self, matrix):
            n = len(matrix[0]) if matrix else 0
            return SolveResult(True, list(range(n)))

        def solve_all(self, matrix):
            n = len(matrix[0]) if matrix else 0
            return [list(range(n))]

    solver = get("dummy_identity")
    assert solver.solve([[2, 2]]).permutation == [0, 1]
