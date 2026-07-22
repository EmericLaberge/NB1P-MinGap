#!/usr/bin/env python3
"""Benchmark SAT solver for NB1P decision problem.

Measures CNF construction time and SAT solving time on:
  1. Random ternary matrices (mostly UNSAT)
  2. Guaranteed-satisfiable matrices (SAT)
"""

from __future__ import annotations

import random
import time

from nb1p import validate_ternary
from nb1p._sat import _build_cnf
from pysat.solvers import Solver


def random_matrix(m: int, n: int, seed: int) -> list[list[int]]:
    """Generate a random ternary matrix."""
    rng = random.Random(seed)
    return [
        [rng.choices([0, 1, 2], weights=[3, 3, 4])[0] for _ in range(n)]
        for _ in range(m)
    ]


def sat_matrix(m: int, n: int, seed: int) -> list[list[int]]:
    """Generate a ternary matrix guaranteed to satisfy NB1P.

    Strategy: pick a random permutation π, then for each row, place 1s
    in a contiguous span under π. Columns inside the span get 1 or 2;
    columns outside get 0 or 2.
    """
    rng = random.Random(seed)
    pi = list(range(n))
    rng.shuffle(pi)

    matrix = []
    for _ in range(m):
        # Pick a random span [L, R] in position space
        span_len = rng.randint(2, max(2, n // 2))
        L = rng.randint(0, n - span_len)
        R = L + span_len - 1

        cols_in_span = [pi[p] for p in range(L, R + 1)]
        cols_outside = [pi[p] for p in range(n) if p < L or p > R]

        row = [0] * n
        # Inside span: 1 or 2 (at least two 1s)
        n_ones = rng.randint(2, max(2, len(cols_in_span)))
        ones = rng.sample(cols_in_span, min(n_ones, len(cols_in_span)))
        for j in cols_in_span:
            row[j] = 1 if j in ones else 2
        # Outside span: 0 or 2
        for j in cols_outside:
            row[j] = rng.choice([0, 2])

        matrix.append(row)
    return matrix


def benchmark(matrix: list[list[int]]) -> dict:
    """Time CNF construction and SAT solving."""
    _, n = validate_ternary(matrix)

    t0 = time.perf_counter()
    cnf = _build_cnf(matrix, n)
    t_build = time.perf_counter() - t0

    n_vars = n * (n - 1) // 2
    n_clauses = len(cnf.clauses)

    t0 = time.perf_counter()
    with Solver(name="cadical153", bootstrap_with=cnf) as solver:
        sat = solver.solve()
    t_solve = time.perf_counter() - t0

    return {
        "n_clauses": n_clauses,
        "t_build_ms": t_build * 1000,
        "t_solve_ms": t_solve * 1000,
        "t_total_ms": (t_build + t_solve) * 1000,
        "sat": sat,
    }


def run_suite(label: str, gen_fn, sizes_n: list[int], k_rows: list[int], n_trials: int):
    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    print(
        f"{'n':>4} {'m':>4} {'clauses':>8} {'build_ms':>10} {'solve_ms':>10} "
        f"{'total_ms':>10} {'sat%':>5}"
    )
    print("-" * 60)

    for n in sizes_n:
        for k in k_rows:
            m = k * n
            results = []
            for trial in range(n_trials):
                seed = n * 1000 + k * 100 + trial
                matrix = gen_fn(m, n, seed)
                results.append(benchmark(matrix))

            avg_clauses = sum(r["n_clauses"] for r in results) / n_trials
            avg_build = sum(r["t_build_ms"] for r in results) / n_trials
            avg_solve = sum(r["t_solve_ms"] for r in results) / n_trials
            avg_total = sum(r["t_total_ms"] for r in results) / n_trials
            sat_pct = sum(1 for r in results if r["sat"]) / n_trials * 100

            print(
                f"{n:>4} {m:>4} {avg_clauses:>8.0f} {avg_build:>10.1f} "
                f"{avg_solve:>10.1f} {avg_total:>10.1f} {sat_pct:>4.0f}%"
            )


def main():
    sizes = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    k_rows = [1, 2, 5]
    n_trials = 1000

    run_suite("RANDOM MATRICES (mostly UNSAT)", random_matrix, sizes, k_rows, n_trials)
    run_suite(
        "SATISFIABLE MATRICES (guaranteed SAT)", sat_matrix, sizes, k_rows, n_trials
    )


if __name__ == "__main__":
    main()
