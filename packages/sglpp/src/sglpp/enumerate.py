"""Enumeration of Dollo-consistent loss assignments over tree edges."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, Generator, List

from sglpp.tree import EdgeInfo


def enumerate_scenarios(
    edges: List[EdgeInfo], n_genes: int
) -> Generator[Dict[int, FrozenSet[int]], None, None]:
    """Every Dollo-consistent loss assignment (edge_id -> frozenset)."""
    return _walk_all(edges, n_genes)


def _walk_all(
    edges: List[EdgeInfo], n_genes: int
) -> Generator[Dict[int, FrozenSet[int]], None, None]:
    all_g = frozenset(range(n_genes))

    def present(eid: int, losses: Dict[int, FrozenSet[int]]) -> FrozenSet[int]:
        lost = frozenset()
        for a in edges[eid].ancestors:
            lost |= losses.get(a, frozenset())
        return all_g - lost

    def walk(idx: int, losses: Dict[int, FrozenSet[int]]):
        if idx >= len(edges):
            yield dict(losses)
            return
        eid = edges[idx].id
        present_here = sorted(present(eid, losses))
        for sz in range(len(present_here) + 1):
            for tup in combinations(present_here, sz):
                losses[eid] = frozenset(tup)
                yield from walk(idx + 1, losses)
        del losses[eid]

    return walk(0, {})


def _present_at(
    eid: int,
    edges: List[EdgeInfo],
    losses: Dict[int, FrozenSet[int]],
    n_genes: int,
) -> FrozenSet[int]:
    lost: FrozenSet[int] = frozenset()
    for a in edges[eid].ancestors:
        lost |= losses.get(a, frozenset())
    return frozenset(range(n_genes)) - lost


def _is_constraining(loss: FrozenSet[int], present: FrozenSet[int]) -> bool:
    return len(loss) >= 2 and len(present - loss) >= 1


def enumerate_scenarios_pruned(
    edges: List[EdgeInfo],
    n_genes: int,
    min_constraining: int = 2,
    max_total_losses: int = 99,
) -> Generator[Dict[int, FrozenSet[int]], None, None]:
    """Dollo assignments with >= min_constraining constraining rows."""
    return _walk_pruned(edges, n_genes, min_constraining, max_total_losses)


def _walk_pruned(
    edges: List[EdgeInfo],
    n_genes: int,
    min_constraining: int,
    max_total_losses: int,
) -> Generator[Dict[int, FrozenSet[int]], None, None]:
    n_edges = len(edges)

    def walk(
        idx: int,
        losses: Dict[int, FrozenSet[int]],
        n_constr: int,
        total_loss: int,
    ):
        if idx >= n_edges:
            if n_constr >= min_constraining:
                yield dict(losses)
            return

        remaining = n_edges - idx
        if n_constr + remaining < min_constraining:
            return

        eid = edges[idx].id
        present_here = sorted(_present_at(eid, edges, losses, n_genes))

        losses[eid] = frozenset()
        yield from walk(idx + 1, losses, n_constr, total_loss)

        for sz in range(1, len(present_here) + 1):
            if total_loss + sz > max_total_losses:
                break
            for tup in combinations(present_here, sz):
                loss_set = frozenset(tup)
                losses[eid] = loss_set
                new_constr = n_constr
                if _is_constraining(loss_set, frozenset(present_here)):
                    new_constr += 1
                yield from walk(idx + 1, losses, new_constr, total_loss + sz)

        del losses[eid]

    return walk(0, {}, 0, 0)
