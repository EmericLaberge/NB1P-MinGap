"""Dependency-free NB1P decision via backtracking.

The lightweight pure-Python procedure used by ``nb1p.check``.
No SAT/ILP/CP backend needed. NB1P is NP-complete, so the worst case is
exponential; *max_calls* bounds the effort.

Strategy
--------
We build the permutation left to right, one position at a time. At each
position we try every not-yet-placed column. After tentatively placing
a column we run two **incremental feasibility checks**:

1. **Placing a 1** — the column extends the ``[leftmost_1, rightmost_1]``
   span of every constraining row that has a ``1`` in it. We reject the
   placement if a ``0``-column of the same row is already inside that
   span — a ``0`` trapped between two ``1``s violates segmentality.

2. **Placing a 0** — the column might land inside the existing span of
   ones in a constraining row where it's a ``0``. Same violation.

If both checks pass, we recurse. On failure we undo the span bookkeeping
and try the next candidate column.
"""

from __future__ import annotations

from itertools import permutations

from nb1p.types import Matrix, Permutation


def check_nb1p(
    matrix: Matrix, n_cols: int, max_calls: int = 5_000_000
) -> tuple[bool, Permutation | None]:
    """Decide NB1P via backtracking.

    A row is *segmental* iff no ``0`` appears between two ``1``s
    (``2`` = wildcard / already lost by an ancestor).

    Returns:
        ``(holds, permutation_or_None)``. If the call budget is exhausted
        before a decision, returns ``(True, None)`` (undetermined — treated
        as "not proven unsatisfiable").
    """
    n = n_cols
    if not matrix:
        return True, list(range(n))

    # ── Extract constraining rows ──────────────────────────────────────
    # A row constrains the permutation only if it has ≥ 2 ones: a single
    # one can never trap a zero between two ones.
    constraints: list[tuple[frozenset[int], frozenset[int]]] = []
    for row in matrix:
        ones = frozenset(j for j in range(n) if row[j] == 1)
        zeros = frozenset(j for j in range(n) if row[j] == 0)
        if len(ones) >= 2:
            constraints.append((ones, zeros))

    if not constraints:
        return True, list(range(n))

    # ── Cross-index: for each column, which constraints see it as 1 / 0 ──
    col_as_one: list[list[int]] = [[] for _ in range(n)]
    col_as_zero: list[list[int]] = [[] for _ in range(n)]
    for ci, (ones, zeros) in enumerate(constraints):
        for j in ones:
            col_as_one[j].append(ci)
        for j in zeros:
            col_as_zero[j].append(ci)

    # ── Incremental span trackers (one set per constraint) ─────────────
    n_constraints = len(constraints)
    leftmost_one = [n] * n_constraints      # smallest position of a placed 1
    rightmost_one = [-1] * n_constraints    # largest position of a placed 1
    ones_count = [0] * n_constraints        # how many 1s placed so far
    placed: dict[int, int] = {}             # column → its position
    call_count = [0]

    def _search(position: int) -> bool:
        """Fill *position* and beyond; return True on success."""
        if position == n:
            return True
        call_count[0] += 1
        if call_count[0] > max_calls:
            return False

        for col in range(n):
            if col in placed:
                continue
            placed[col] = position

            # Snapshot the span state for undo on failure.
            undo: list[tuple[int, int, int, int]] = []
            feasible = True

            # Check 1: extending the span of rows where this column is a 1.
            # After extending, reject if a zero of the same row sits inside.
            for ci in col_as_one[col]:
                undo.append(
                    (ci, leftmost_one[ci], rightmost_one[ci], ones_count[ci])
                )
                ones_count[ci] += 1
                if position < leftmost_one[ci]:
                    leftmost_one[ci] = position
                if position > rightmost_one[ci]:
                    rightmost_one[ci] = position
                if ones_count[ci] >= 2:
                    lo, hi = leftmost_one[ci], rightmost_one[ci]
                    for zero_col in constraints[ci][1]:
                        if zero_col in placed and lo < placed[zero_col] < hi:
                            feasible = False
                            break
                if not feasible:
                    break

            # Check 2: this column is a 0 in some rows — does it land
            # inside the existing span of ones?
            if feasible:
                for ci in col_as_zero[col]:
                    if ones_count[ci] >= 2:
                        lo, hi = leftmost_one[ci], rightmost_one[ci]
                        if lo < position < hi:
                            feasible = False
                            break

            if feasible and _search(position + 1):
                return True

            # Undo: restore span bookkeeping for every constraint we touched.
            for ci, old_lo, old_hi, old_cnt in undo:
                ones_count[ci] = old_cnt
                leftmost_one[ci] = old_lo
                rightmost_one[ci] = old_hi
            del placed[col]

        return False

    if _search(0):
        perm = [0] * n
        for col, pos in placed.items():
            perm[pos] = col
        return True, perm
    if call_count[0] >= max_calls:
        return True, None
    return False, None


def check_nb1p_fast(
    matrix: Matrix, n_cols: int
) -> tuple[bool, Permutation | None]:
    """Decide NB1P via full permutation enumeration (practical for n ≤ 8).

    A row is segmental iff no ``0`` appears between any two ``1``s.
    ``2``s (ancestor losses) are wildcards — they don't break contiguity.
    """
    if not matrix:
        return True, list(range(n_cols))

    constraints: list[tuple[list[int], frozenset[int]]] = []
    for row in matrix:
        ones = [j for j in range(n_cols) if row[j] == 1]
        zeros = frozenset(j for j in range(n_cols) if row[j] == 0)
        if len(ones) >= 2:
            constraints.append((ones, zeros))

    if not constraints:
        return True, list(range(n_cols))

    for perm in permutations(range(n_cols)):
        pos = {col: idx for idx, col in enumerate(perm)}
        feasible = True
        for ones, zeros in constraints:
            positions = sorted(pos[j] for j in ones)
            lo, hi = positions[0], positions[-1]
            for z in zeros:
                if lo < pos[z] < hi:
                    feasible = False
                    break
            if not feasible:
                break
        if feasible:
            return True, list(perm)

    return False, None
