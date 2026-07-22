"""Ordering-variable encoding shared by the NB1P SAT and ILP backends.
One boolean variable v(a,b) per column pair (0 ≤ a < b < n): true when
column a comes before column b.
SAT backends use 1-indexed variable IDs (PySAT convention);
ILP backends use 0-indexed (HiGHS convention).
Exports: pair↔index mapping, transitivity triples, NB1P constraint
triplets (j1, j2, z), linear row builder for ILP, and permutation
extraction from solved variable values.
"""

from __future__ import annotations

from functools import cmp_to_key

from nb1p.types import Matrix, Permutation


def num_pairs(n: int) -> int:
    """Number of ordering variables for ``n`` columns: ``n(n-1)/2``."""
    return n * (n - 1) // 2


def pair_index(a: int, b: int, n: int) -> int:
    """0-indexed variable index for the pair ``(a, b)`` with ``a < b``.

    Maps the upper triangle of the ``n×n`` comparison matrix to
    ``[0, n(n-1)/2)``. Inverse of the SAT solver's ``vid()`` minus 1.

    Derivation: row ``a`` of the upper triangle has ``n - a - 1`` entries.
    The cumulative offset of row ``a`` is ``a(2n - a - 1) / 2``; the column
    offset within that row is ``b - a - 1``.
    """
    return a * (2 * n - a - 1) // 2 + (b - a) - 1

def transitivity_triples(n: int):
    """Yield ``(ab, bc, ac)`` variable-index triples for transitivity.

    For each triple ``a < b < c``::

        o[a,b] + o[b,c] - 1 ≤ o[a,c]     (if a<b and b<c then a<c)
        o[a,c] ≤ o[a,b] + o[b,c]          (converse)
    """
    for a in range(n - 2):
        for b in range(a + 1, n - 1):
            ab = pair_index(a, b, n)
            for c in range(b + 1, n):
                bc = pair_index(b, c, n)
                ac = pair_index(a, c, n)
                yield ab, bc, ac


def nb1p_triplets(matrix: Matrix, n: int):
    """Yield ``(j1, j2, z)`` for every NB1P constraint across all rows.

    For each row, each pair ``(j1, j2)`` of 1-columns, each 0-column ``z``::

        before(z, j1) == before(z, j2)

    This encodes segmentality: if a zero-column ``z`` were on one side of
    ``j1`` but the other side of ``j2``, it would land between two ones,
    breaking the segmental property.
    """
    for row in matrix:
        ones = [j for j in range(n) if row[j] == 1]
        zeros = [j for j in range(n) if row[j] == 0]
        if len(ones) < 2:
            continue
        for i1 in range(len(ones)):
            for i2 in range(i1 + 1, len(ones)):
                j1, j2 = ones[i1], ones[i2]
                for z in zeros:
                    yield j1, j2, z


def is_before_from(var_val: dict[int, float], a: int, b: int, n: int,
                   one_indexed: bool = False) -> bool:
    """Read ordering from variable values.

    Args:
        var_val: dict mapping ``pair_index`` (or ``pair_index + 1`` if
            *one_indexed*) to its truth/value.
        one_indexed: ``True`` for SAT (vid starts at 1), ``False`` for ILP.
    """
    if a < b:
        idx = pair_index(a, b, n) + (1 if one_indexed else 0)
        return bool(var_val.get(idx, False))
    idx = pair_index(b, a, n) + (1 if one_indexed else 0)
    return not bool(var_val.get(idx, False))


def extract_perm(var_val: dict[int, float], n: int,
                 one_indexed: bool = False) -> Permutation:
    """Extract a permutation from order-variable assignments.

    ``perm[k]`` = column at position ``k``.
    """
    def _before(a, b):
        return is_before_from(var_val, a, b, n, one_indexed)

    return sorted(range(n), key=cmp_to_key(
        lambda a, b: -1 if _before(a, b) else 1))
