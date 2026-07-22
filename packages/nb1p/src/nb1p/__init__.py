"""NB1P — Non-Breaking Ones Property decision problem."""

from __future__ import annotations

from nb1p._check import check_nb1p, check_nb1p_fast
from nb1p._registry import available, get, register
from nb1p.encoding import (
    extract_perm,
    is_before_from,
    nb1p_triplets,
    num_pairs,
    pair_index,
    transitivity_triples,
)
from nb1p.matrices import paper_example, random_matrix
from nb1p.registry import SolverRegistry
from nb1p.ternary import validate_ternary
from nb1p.types import Matrix, NB1PSolver, Permutation, SolveResult
from nb1p.verify import verify

__all__ = [
    "Matrix", "Permutation", "SolveResult", "NB1PSolver",
    "SolverRegistry", "validate_ternary", "verify",
    "extract_perm", "is_before_from", "pair_index", "num_pairs",
    "transitivity_triples", "nb1p_triplets",
    "paper_example", "random_matrix",
    "check", "solve", "solve_all", "check_nb1p", "check_nb1p_fast",
    "available", "register", "get",
]


def solve(matrix: Matrix, *, solver: str = "auto") -> SolveResult:
    return get(solver).solve(matrix)


def solve_all(matrix: Matrix, *, solver: str = "auto") -> list[Permutation]:
    return get(solver).solve_all(matrix)


def check(matrix: Matrix, *, max_calls: int = 5_000_000) -> bool:
    _, n = validate_ternary(matrix)
    holds, _perm = check_nb1p(matrix, n, max_calls=max_calls)
    return holds
