"""Verify the reduction NB1P ⟹ MinGapOrdering-Decision(k=0).

For every test matrix M, we check::

    NB1P(M) = YES  ⟺  min_cost(M) = 0.

This validates the key lemma underlying the theorem that
MinGapOrdering-Decision is NP-complete: ``cost(π*) = 0  ⟺  NB1P holds``.

Usage:
    python scripts/verify_reduction.py
"""

from __future__ import annotations

import itertools
import random
import sys

import mingap
import nb1p
from mingap.score import block_cost


def nonbetweenness_matrix(triples, n) -> list[list[int]]:
    """Build the ternary matrix from the NB1P NP-hardness reduction."""
    matrix = []
    for p, q, r in triples:
        row = [2] * n
        row[p] = 1
        row[r] = 1
        row[q] = 0
        matrix.append(row)
    return matrix


def all_matrices_small(m: int, n: int):
    """Enumerate all ternary matrices of size m×n (tiny sizes only)."""
    rows_pool = list(itertools.product([0, 1, 2], repeat=n))
    return [list(list(r) for r in combo)
            for combo in itertools.product(rows_pool, repeat=m)]


def check_one(matrix: list[list[int]], label: str) -> bool:
    nb1p_res = nb1p.solve(matrix, solver="brute_force")
    opt = mingap.solve(matrix, solver="brute_force")
    if opt.permutation is not None:
        verified_cost = block_cost(matrix, opt.permutation)
        assert verified_cost == opt.cost, (
            f"{label}: gap_count={opt.cost} but standalone={verified_cost}")

    ok = (nb1p_res.satisfiable == (opt.cost == 0))
    status = "✓" if ok else "✗ MISMATCH"
    print(f"  {status}  {label:40s}  NB1P={nb1p_res.satisfiable!s:5s}  min_gap={opt.cost}")
    if not ok:
        print(f"    COUNTEREXAMPLE: matrix = {matrix}")
        print(f"    best_perm = {opt.permutation}")
    return ok


def main() -> None:
    all_ok = True
    count = 0

    print("=" * 72)
    print("Reduction verification: NB1P(M)=YES  ⟺  min_cost(M)=0")
    print("=" * 72)

    print("\n[1] Exhaustive: all 3×3 ternary matrices (27^3 = 19683)")
    for matrix in all_matrices_small(3, 3):
        count += 1
        if not check_one(matrix, f"3×3 #{count}"):
            all_ok = False
    print(f"    Tested {count} matrices.")

    count = 0
    print("\n[2] Exhaustive: all 2×4 ternary matrices (81^2 = 6561)")
    for matrix in all_matrices_small(2, 4):
        count += 1
        if not check_one(matrix, f"2×4 #{count}"):
            all_ok = False
    print(f"    Tested {count} matrices.")

    print("\n[3] Paper example (6×5)")
    if not check_one(nb1p.paper_example(), "paper_example"):
        all_ok = False

    print("\n[4] Non-Betweenness reduction matrices")
    test_triples = [
        ([(0, 1, 2)], 3),
        ([(0, 1, 2), (1, 2, 3)], 4),
        ([(0, 1, 2), (0, 2, 3), (1, 3, 4)], 5),
        ([(0, 1, 2), (0, 3, 2), (1, 3, 4)], 5),
        ([(0, 2, 4), (1, 3, 5), (2, 4, 6), (3, 5, 7),
          (0, 4, 7), (1, 5, 6)], 8),
    ]
    for i, (triples, n) in enumerate(test_triples):
        M = nonbetweenness_matrix(triples, n)
        if not check_one(M, f"nonbet #{i} ({len(triples)} triples, n={n})"):
            all_ok = False

    print("\n[5] Random matrices")
    rng = random.Random(42)
    for i in range(200):
        m = rng.randint(2, 6)
        n = rng.randint(3, 7)
        matrix = nb1p.random_matrix(m, n, p1=0.3, p0=0.15, rng=rng)
        if not check_one(matrix, f"random {m}×{n} #{i}"):
            all_ok = False

    print("\n" + "=" * 72)
    if all_ok:
        print("ALL CHECKS PASSED — reduction verified on every instance.")
    else:
        print("MISMATCH FOUND — reduction has a counterexample!")
        sys.exit(1)


if __name__ == "__main__":
    main()
