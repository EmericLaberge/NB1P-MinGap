"""Haddadi–Layouni 1.5-approximation for CBM, adapted to MinGap via TSP/Christofides.

Requires: pip install mingap[tsp] (networkx).

Reduction chain (Haddadi & Layouni 2008, see docs/algo-research.md):
MinGap → Consecutive Block Minimization (CBM) → Circular Block Minimization
(add a dummy all-zero column) → metric TSP on columns → Christofides'
1.5-approximation for metric TSP.

The ternary matrix is collapsed to binary by treating ``2`` (wildcard/glue)
as active — the conservative collapse, since a 2 can always serve as block
glue and never forces a split.
"""

from __future__ import annotations

import networkx as nx

from mingap._registry import register
from mingap.score import block_cost
from mingap.types import Matrix, OptimizeResult
from nb1p import validate_ternary


@register("tsp_approx")
class TSPApproxSolver:
    """Christofides-based heuristic for MinGap via the CBM→TSP reduction.

    Builds a complete graph on the columns plus one dummy all-zero column
    (which makes the block problem circular). The edge weight rewards
    adjacencies that merge 1-blocks across rows::

        w(u, v) = m − |{r : row[u] ∈ {1,2} and row[v] ∈ {1,2}}|

    and the metric TSP tour is approximated with Christofides' algorithm
    (networkx). Removing the dummy from the tour yields a linear column
    permutation; its cost is evaluated with the true ternary
    :func:`mingap.score.block_cost`.

    1.5-approximation for the CBM objective; for MinGap the ratio transfers
    only when the optimum is large relative to m — in the low-cost
    (biological) regime the additive shift between the two objectives means
    no constant-factor guarantee holds (see ``docs/algo-research.md`` §B).

    Deterministic: the graph is built with edges inserted in sorted
    (column, column) order and the extracted tour is canonically oriented
    with the smaller endpoint first (block cost is reversal-invariant).
    Opt-in only; not part of ``auto_order``.
    """

    def solve(self, matrix: Matrix) -> OptimizeResult:
        m, n = validate_ternary(matrix)
        identity = list(range(n))
        if n <= 1:
            return OptimizeResult(identity, block_cost(matrix, identity))
        if m == 0:
            return OptimizeResult(identity, 0)

        # Binary collapse: 2 (wildcard) counts as active, like 1.
        active = [[1 if row[j] in (1, 2) else 0 for j in range(n)] for row in matrix]

        # Vertices 0..n-1 are the columns; vertex n is the dummy all-zero
        # column that turns the linear problem into a circular one.
        dummy = n
        graph = nx.Graph()
        graph.add_nodes_from(range(n + 1))
        for u in range(n):
            for v in range(u + 1, n):
                shared = sum(1 for row in active if row[u] and row[v])
                graph.add_edge(u, v, weight=m - shared)
        for u in range(n):
            graph.add_edge(u, dummy, weight=m)

        tour = nx.approximation.christofides(graph, weight="weight")
        # tour is a closed walk [v0, …, vk-1, v0]; rotate to the dummy and
        # drop it to get the linear permutation.
        tour = tour[:-1]
        i = tour.index(dummy)
        perm = tour[i + 1 :] + tour[:i]
        # Canonical orientation (block cost is reversal-invariant): the
        # smaller endpoint comes first.
        if perm[0] > perm[-1]:
            perm.reverse()

        return OptimizeResult(perm, block_cost(matrix, perm))
