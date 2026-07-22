"""Validation of ternary matrices ``M ∈ {0,1,2}^{m×n}``."""

from __future__ import annotations

from nb1p.types import Matrix

TERNARY_VALUES = frozenset({0, 1, 2})


def validate_ternary(matrix: Matrix) -> tuple[int, int]:
    """Check *matrix* is a valid ternary matrix.
    Rows must all be the same length, and every cell in {0, 1, 2}.
    Semantics: 1 = lost, 0 = present, 2 = ancestral (wildcard).
    Anything outside {0,1,2} is rejected outright — we don't silently
    treat garbage as wildcard.
    Args:
        matrix: m×n matrix (list of rows).
    Returns:
        ``(m, n)`` — the validated dimensions.
    Raises:
        ValueError: ragged matrix or non-ternary cell.
    """
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError(
                f"row {i} has length {len(row)}, expected {n} "
                f"(matrix must be rectangular)")
        for j, cell in enumerate(row):
            if cell not in TERNARY_VALUES:
                raise ValueError(
                    f"cell ({i},{j}) = {cell!r} is not ternary; "
                    f"expected one of {{0, 1, 2}}")
    return m, n
