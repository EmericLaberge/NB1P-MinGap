"""Generate distinct unlabelled rooted binary tree shapes."""

from __future__ import annotations

from typing import Generator, List

from sglpp.tree import PhyloTree


def generate_topologies(n_leaves: int) -> List[PhyloTree]:
    """All distinct *unlabelled* rooted binary tree shapes with n_leaves."""
    if n_leaves == 1:
        return [PhyloTree.leaf(0)]
    if n_leaves == 2:
        return [PhyloTree.node(PhyloTree.leaf(0), PhyloTree.leaf(1))]

    seen: set[str] = set()
    result: List[PhyloTree] = []
    for left_n in range(1, n_leaves):
        right_n = n_leaves - left_n
        for lt in _sub_topos(left_n, 0):
            for rt in _sub_topos(right_n, left_n):
                t = PhyloTree.node(lt, rt)
                h = _topo_hash(t)
                if h not in seen:
                    seen.add(h)
                    result.append(t)
    return result


def _sub_topos(n: int, offset: int) -> Generator[PhyloTree, None, None]:
    if n == 1:
        yield PhyloTree.leaf(offset)
        return
    for k in range(1, n):
        for l in _sub_topos(k, offset):
            for r in _sub_topos(n - k, offset + k):
                yield PhyloTree.node(l, r)


def _topo_hash(t: PhyloTree) -> str:
    if t.is_leaf():
        return "L"
    lh, rh = _topo_hash(t.left), _topo_hash(t.right)
    return f"({min(lh, rh)},{max(lh, rh)})"
