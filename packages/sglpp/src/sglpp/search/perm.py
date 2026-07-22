"""Permutation search helpers used by the NB1P counter-example visualizer.

Pure enumeration — for small ``n`` (≤ 8 in practice). Moved out of
:mod:`sglpp.viz` so the algorithm isn't hidden inside a drawing
module.
"""

from __future__ import annotations

from itertools import permutations


def best_partial_perm(matrix: list[list[int]], n: int) -> list[int] | None:
    """Find a permutation of ``range(n)`` minimising the number of violated
    constraining rows.  A "best attempt" for counter-example visualization
    when no perfect permutation exists.

    A row *violates* a permutation if it has ≥2 ones and at least one 0
    lies strictly between the leftmost and rightmost 1 in the permuted row.
    """
    constraining = []
    for row in matrix:
        ones = [j for j in range(n) if row[j] == 1]
        zeros = frozenset(j for j in range(n) if row[j] == 0)
        if len(ones) >= 2:
            constraining.append((ones, zeros))

    if not constraining:
        return list(range(n))

    best: list[int] | None = None
    best_violations = len(constraining) + 1
    for perm in permutations(range(n)):
        pos = {c: i for i, c in enumerate(perm)}
        violations = 0
        for ones, zeros in constraining:
            ps = sorted(pos[j] for j in ones)
            lo, hi = ps[0], ps[-1]
            for z in zeros:
                if lo < pos[z] < hi:
                    violations += 1
                    break
        if violations < best_violations:
            best_violations = violations
            best = list(perm)
            if violations == 0:
                return best

    return best


def violations_of(perm: list[int], matrix: list[list[int]], n: int) -> list[int]:
    """Return the row-indices violated by *perm* (rows with ≥2 ones whose
    span contains a 0 in the permuted order).
    """
    pos = {c: i for i, c in enumerate(perm)}
    violated: list[int] = []
    for i, row in enumerate(matrix):
        ones = [j for j in range(n) if row[j] == 1]
        if len(ones) < 2:
            continue
        zeros = frozenset(j for j in range(n) if row[j] == 0)
        ps = sorted(pos[j] for j in ones)
        lo, hi = ps[0], ps[-1]
        for z in zeros:
            if lo < pos[z] < hi:
                violated.append(i)
                break
    return violated


__all__ = ["best_partial_perm", "violations_of"]
