"""SAT-based NB1P backend using CaDiCaL via PySAT.
Ordering variables v(a,b) for 0 ≤ a < b < n.
v(a,b) = True iff column a comes before column b.
Clauses:
  1. Transitivity — 2 clauses per triple a < b < c.
  2. NB1P — 2 binary clauses per (j1, j2, z).
Requires the sat extra (pip install mininv[sat]).
"""

from __future__ import annotations

from pysat.formula import CNF
from pysat.solvers import Solver

from nb1p import (
    extract_perm,
    nb1p_triplets,
    pair_index,
    transitivity_triples,
    validate_ternary,
    verify,
)
from nb1p._registry import register
from nb1p.types import Matrix, Permutation, SolveResult


def _before_lit(x: int, y: int, n: int) -> int:
    """PySAT literal for ``v(x, y)`` (1-indexed). Positive iff x < y."""
    if x < y:
        return pair_index(x, y, n) + 1
    return -(pair_index(y, x, n) + 1)


def _build_cnf(matrix: Matrix, n: int) -> CNF:
    """Build the NB1P CNF formula (1-indexed vars, PySAT convention)."""
    cnf = CNF()

    for ab, bc, ac in transitivity_triples(n):
        v_ab, v_bc, v_ac = ab + 1, bc + 1, ac + 1
        cnf.append([-v_ab, -v_bc, v_ac])
        cnf.append([v_ab, v_bc, -v_ac])

    for j1, j2, z in nb1p_triplets(matrix, n):
        lit_before_z_j1 = _before_lit(z, j1, n)
        lit_before_z_j2 = _before_lit(z, j2, n)
        cnf.append([-lit_before_z_j1, lit_before_z_j2])
        cnf.append([lit_before_z_j1, -lit_before_z_j2])

    return cnf


def _model_to_perm(model: list[int], n: int) -> Permutation:
    truth = {abs(v): v > 0 for v in model}
    return extract_perm(truth, n, one_indexed=True)


@register("sat")
class SATSolver:
    """CaDiCaL-backed decision + enumeration."""

    def __init__(self, solver_name: str = "cadical153"):
        self.solver_name = solver_name

    def solve(self, matrix: Matrix) -> SolveResult:
        _, n = validate_ternary(matrix)
        if n <= 1:
            return SolveResult(True, list(range(n)))

        cnf = _build_cnf(matrix, n)
        with Solver(name=self.solver_name, bootstrap_with=cnf) as solver:
            if not solver.solve():
                return SolveResult(False, None)
            perm = _model_to_perm(solver.get_model(), n)

        assert verify(matrix, perm), "SAT returned invalid permutation"
        return SolveResult(True, perm)

    def solve_all(self, matrix: Matrix) -> list[Permutation]:
        """Enumerate ALL valid NB1P permutations using blocking clauses.

        Each solution adds a blocking clause of ``n(n-1)/2`` literals, so
        enumeration slows as solutions accumulate. For small ``n`` brute
        force is often faster for full enumeration.
        """
        _, n = validate_ternary(matrix)
        if n <= 1:
            return [list(range(n))]

        cnf = _build_cnf(matrix, n)
        solutions: list[Permutation] = []

        with Solver(name=self.solver_name, bootstrap_with=cnf) as solver:
            while solver.solve():
                model = solver.get_model()
                perm = _model_to_perm(model, n)
                assert verify(matrix, perm), f"SAT invalid: {perm}"
                solutions.append(perm)

                truth = {abs(v): v > 0 for v in model}
                block = []
                for a in range(n):
                    for b in range(a + 1, n):
                        vid = pair_index(a, b, n) + 1
                        block.append(-vid if truth[vid] else vid)
                solver.add_clause(block)

        return solutions
