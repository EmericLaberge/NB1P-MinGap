"""Phylogenetic tree data structures, parsing, and edge numbering."""

from __future__ import annotations

import ast
from typing import List, Optional, Tuple, Union


class PhyloTree:
    """Rooted binary phylogenetic tree."""

    __slots__ = ("left", "right", "label")

    def __init__(
        self,
        left: Optional["PhyloTree"] = None,
        right: Optional["PhyloTree"] = None,
        label: Optional[int] = None,
    ):
        self.left = left
        self.right = right
        self.label = label

    @staticmethod
    def leaf(label: int) -> "PhyloTree":
        return PhyloTree(label=label)

    @staticmethod
    def node(left: "PhyloTree", right: "PhyloTree") -> "PhyloTree":
        return PhyloTree(left=left, right=right)

    def is_leaf(self) -> bool:
        return self.left is None

    def leaves(self) -> List[int]:
        if self.is_leaf():
            return [self.label]
        return self.left.leaves() + self.right.leaves()

    def n_leaves(self) -> int:
        return len(self.leaves())

    def depth(self) -> int:
        if self.is_leaf():
            return 0
        return 1 + max(self.left.depth(), self.right.depth())

    def __repr__(self):
        if self.is_leaf():
            return str(self.label)
        return f"({self.left},{self.right})"


class EdgeInfo:
    """Metadata for one edge of the tree (parent → child)."""

    __slots__ = ("id", "parent", "child", "ancestors")

    def __init__(
        self, id: int, parent: PhyloTree, child: PhyloTree, ancestors: Tuple[int, ...]
    ):
        self.id = id
        self.parent = parent
        self.child = child
        self.ancestors = ancestors  # ancestor edge-ids (root-ward)

    def __repr__(self):
        return f"Edge({self.id}, anc={self.ancestors})"


def number_edges(tree: PhyloTree) -> List[EdgeInfo]:
    """Pre-order DFS edge numbering.  O(n_leaves)."""
    edges: List[EdgeInfo] = []
    counter = [0]

    def _dfs(node: PhyloTree, anc: Tuple[int, ...]):
        if node.is_leaf():
            return
        eid = counter[0]
        counter[0] += 1
        edges.append(EdgeInfo(eid, node, node.left, anc))
        _dfs(node.left, anc + (eid,))
        eid = counter[0]
        counter[0] += 1
        edges.append(EdgeInfo(eid, node, node.right, anc))
        _dfs(node.right, anc + (eid,))

    _dfs(tree, ())
    return edges


def parse_tree(source: Union[str, tuple, int]) -> PhyloTree:
    """Build a PhyloTree from a nested-tuple representation.

    Accepts either a string ``"(0,(1,2))"`` or the already-parsed
    nested-tuple structure ``(0, (1, 2))`` produced by ``ast.literal_eval``
    or ``json.loads``.  Leaves are integers; internal nodes are 2-tuples.
    """
    if isinstance(source, str):
        source = ast.literal_eval(source.strip())
    if isinstance(source, int):
        return PhyloTree.leaf(source)
    left, right = source
    return PhyloTree.node(parse_tree(left), parse_tree(right))
