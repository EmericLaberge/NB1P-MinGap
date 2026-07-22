"""Rollout greedy MinGap — K best-fit-insertion variants, best result wins.

Each variant runs the same O(n³) best-fit insertion as :class:`GreedySolver`,
but with a different initial column-priority order, which controls both the
tie-breaking among equal-cost (column, position) candidates and which columns
get committed first. Strategies:

- 0: identity order (exactly reproduces the plain ``greedy`` baseline),
- 1: descending 1-count per column,
- 2: descending 2-count per column,
- 3: descending column blockiness (number of active {1,2} runs down the
  column),
- 4..K-1: shuffled orders from ``random.Random(strategy * 7919)``.

The minimum-cost result across strategies is returned, so the rollout is
never worse than the single greedy run (strategy 0). Deterministic: fixed
seeds, and ``min`` keeps the lowest strategy index on cost ties.

Variants run in parallel with :class:`concurrent.futures.ProcessPoolExecutor`
(``n_jobs = min(n_jobs, os.cpu_count(), k)``). Opt-in only; not part of
``auto_order``. No optional dependencies.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor

from mingap._registry import register
from mingap.score import block_cost
from mingap.types import Matrix, OptimizeResult
from nb1p import validate_ternary

#: Prime stride for the per-strategy shuffle seeds (same constant as the
#: bench seed derivation, so rollout seeds never collide with matrix seeds).
_SEED_STRIDE = 7919


def _column_blockiness(matrix: Matrix, j: int) -> int:
    """Number of maximal active ({1,2}) runs down column *j*."""
    blocks = 0
    in_block = False
    for row in matrix:
        if row[j] in (1, 2):
            blocks += not in_block
            in_block = True
        else:
            in_block = False
    return blocks


def _strategy_order(matrix: Matrix, n: int, strategy: int) -> list[int]:
    """Column-priority order for rollout strategy *strategy*."""
    cols = list(range(n))
    if strategy == 0:
        return cols  # baseline: index order, identical to GreedySolver
    if strategy == 1:  # descending number of 1s per column
        key = lambda j: sum(1 for row in matrix if row[j] == 1)  # noqa: E731
    elif strategy == 2:  # descending number of 2s (glue) per column
        key = lambda j: sum(1 for row in matrix if row[j] == 2)  # noqa: E731
    elif strategy == 3:  # descending column blockiness
        key = lambda j: _column_blockiness(matrix, j)
    else:  # strategy >= 4: shuffled order with a fixed per-strategy seed
        random.Random(strategy * _SEED_STRIDE).shuffle(cols)
        return cols
    # Descending key; column index breaks ties (deterministic).
    return sorted(cols, key=lambda j: (-key(j), j))


def _greedy_with_order(matrix: Matrix, priority: list[int]) -> tuple[int, list[int]]:
    """Best-fit insertion with ties broken by *priority* order, then slot.

    With ``priority == list(range(n))`` this is step-for-step identical to
    :class:`mingap._greedy.GreedySolver`.
    """
    remaining = list(priority)
    order: list[int] = []
    for _ in range(len(priority)):
        best_cost: int | None = None
        best_col = -1
        best_pos = -1
        k = len(order) + 1
        for col in remaining:
            for pos in range(k):
                cand = order[:pos] + [col] + order[pos:]
                c = block_cost(matrix, cand)
                if best_cost is None or c < best_cost:
                    best_cost = c
                    best_col = col
                    best_pos = pos
        order.insert(best_pos, best_col)
        remaining.remove(best_col)
        if best_cost == 0:
            # The rest are pure glue — any order keeps cost 0.
            order.extend(remaining)
            break
    return block_cost(matrix, order), order


def _run_strategy(payload: tuple[Matrix, int, int]) -> tuple[int, list[int]]:
    """Pool worker: run one rollout strategy, return (cost, permutation)."""
    matrix, n, strategy = payload
    return _greedy_with_order(matrix, _strategy_order(matrix, n, strategy))


@register("greedy_rollout")
class GreedyRolloutSolver:
    """Multi-strategy greedy rollout. Runs K greedy variants in parallel,
    returns the best result. Strategies differ in insertion ordering,
    tie-breaking, and starting column choice (see module docstring).
    """

    def __init__(self, k: int = 32, n_jobs: int = 4):
        """k=32 is the knee of the quality curve on the tsp_vs_exact bench
        (.tmp/rollout_k_sweep.py): optima found 10/45 (k=1) -> 19 (k=8) ->
        24 (k=16) -> 27 (k=32) -> 29 (k=64), wall time flat under 4 workers.
        """
        self.k = k
        self.n_jobs = n_jobs

    def solve(self, matrix: Matrix) -> OptimizeResult:
        m, n = validate_ternary(matrix)
        if n == 0:
            return OptimizeResult([], 0)
        if n == 1:
            return OptimizeResult([0], 0)
        if m == 0:
            # No rows → cost is always 0, any permutation is optimal.
            return OptimizeResult(list(range(n)), 0)

        k = max(1, self.k)
        n_jobs = max(1, min(self.n_jobs, os.cpu_count() or 1, k))
        payloads = [(matrix, n, s) for s in range(k)]
        if n_jobs == 1:
            results = [_run_strategy(p) for p in payloads]
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                results = list(pool.map(_run_strategy, payloads))
        # First minimum in strategy order: deterministic, and strategy 0 is
        # the plain greedy baseline so the rollout is never worse than it.
        best_cost, best_perm = min(results, key=lambda r: r[0])
        return OptimizeResult(best_perm, best_cost)
