"""Greedy heuristic MinGap — best-fit insertion baseline.

Builds the column permutation one column at a time. At each step, every
remaining column is tried at every insertion slot in the current partial
order; the (column, position) pair that minimises the resulting block
cost is committed. Runs in O(m n³) cost-function calls (≈ n(n+1)(n+2)/6)
— fast for n ≤ 50 with the numba-accelerated block_cost. Opt-in only;
not added to ``auto_order`` (heuristics are explicit, exact solvers are
auto). No optional dependencies.
"""

from __future__ import annotations

from mingap._registry import register
from mingap.score import block_cost
from mingap.types import Matrix, OptimizeResult
from nb1p import validate_ternary


@register("greedy")
class GreedySolver:
    """Best-fit insertion greedy for the MinGap objective.

    At each step k, for every not-yet-placed column c and every
    insertion position p in ``range(len(order) + 1)``, evaluate
    ``block_cost(matrix, order[:p] + [c] + order[p:])``. Commit the
    pair that yields the lowest cost (ties broken by lowest column
    index, then lowest position — both deterministic on iteration
    order, no random tiebreak needed).

    The 2-as-glue semantics are respected implicitly: columns with
    many 2s and few 1s cost less to place anywhere, so they end up
    parked cheaply while 1-bearing columns are slotted in to suppress
    block splits.
    """

    def solve(self, matrix: Matrix) -> OptimizeResult:
        m, n = validate_ternary(matrix)
        if n == 0:
            return OptimizeResult([], 0)
        if n == 1:
            return OptimizeResult([0], 0)
        if m == 0:
            # No rows → cost is always 0, any permutation is optimal.
            return OptimizeResult(list(range(n)), 0)

        remaining = list(range(n))
        order: list[int] = []
        # Indices sorted for deterministic tiebreaks (lowest column first).
        for _ in range(n):
            best_cost: int | None = None
            best_col = -1
            best_pos = -1
            k = len(order) + 1
            for col in remaining:
                # Build partial candidates incrementally to avoid full
                # list slicing on the hot path.
                for pos in range(k):
                    cand = order[:pos] + [col] + order[pos:]
                    c = block_cost(matrix, cand)
                    if best_cost is None or c < best_cost:
                        best_cost = c
                        best_col = col
                        best_pos = pos
                    # (no early exit: c == 0 doesn't break out cleanly without state; the
                    # outer loop terminates via best_cost == 0 at the end of an iteration anyway.)
            order.insert(best_pos, best_col)
            remaining.remove(best_col)
            if best_cost == 0:
                # Once we hit zero, the rest are pure glue — any order
                # of remaining columns keeps cost 0.
                order.extend(remaining)

                break

        return OptimizeResult(order, block_cost(matrix, order))
