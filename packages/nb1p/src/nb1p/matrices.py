"""Test matrix generators for NB1P / MinGap."""

from __future__ import annotations

import random

from nb1p.types import Matrix


def paper_example() -> Matrix:
    """The 6×5 ternary matrix from the NB1P paper (12 valid permutations)."""
    return [
        [0, 1, 0, 1, 0],
        [1, 2, 0, 2, 1],
        [0, 2, 1, 2, 0],
        [1, 0, 0, 0, 1],
        [2, 1, 1, 0, 2],
        [2, 0, 0, 1, 2],
    ]


def random_matrix(m: int, n: int, p1: float, p0: float,
                  rng: random.Random | None = None) -> Matrix:
    """Generate a random ternary matrix ``M ∈ {0,1,2}^{m×n}``.

    Each cell: ``1`` with prob *p1*, ``0`` with prob *p0*, ``2`` with prob
    ``(1 - p1 - p0)``.
    """
    rng = rng or random
    matrix: Matrix = []
    for _ in range(m):
        row = []
        for _ in range(n):
            r = rng.random()
            if r < p1:
                row.append(1)
            elif r < p1 + p0:
                row.append(0)
            else:
                row.append(2)
        matrix.append(row)
    return matrix
