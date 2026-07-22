"""Recursive cladogram layout for rooted binary phylogenetic trees.

Originally lived in ``app/_tree_layout.py``; moved here so the annotated
renderer in :mod:`sglpp.annotated_tree` and any future tree-rendering
tool can reuse the same coordinates.  ``app/_tree_layout.py`` now re-exports
:class:`TreeLayout` from this module so the Streamlit pages keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

from sglpp.tree import PhyloTree


@dataclass
class TreeLayout:
    """Recursive cladogram layout for a rooted binary tree.

    Attributes
    ----------
    positions : dict[int, tuple[float, float]]
        ``id(node) → (x, y)`` after :meth:`build`.  *x* is a unitless
        leaf-index (0, 1, …, n_leaves-1); *y* is the depth from the
        root.
    leaf_ids : set[int]
        ``id(node)`` of the leaves, in left-to-right order.  Useful to
        distinguish leaves from internal nodes when annotating.
    max_depth : float
        The depth of the deepest node.  Multiplied by ``level_step`` to
        give the figure height.
    n_leaves : int
        Number of leaves.
    level_step : float
        Vertical distance between adjacent depths (default 0.6).
    """

    positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    leaf_ids: Set[int] = field(default_factory=set)
    max_depth: float = 0.0
    n_leaves: int = 0
    level_step: float = 0.6

    @classmethod
    def build(cls, tree: PhyloTree, *, level_step: float = 0.6) -> "TreeLayout":
        """Walk the tree bottom-up and place every node.

        Children are visited in left-to-right order; the parent is placed
        at the x-midpoint of its two children, one level above the deeper
        of the two.  The resulting layout is the ortho-cladogram used
        across the paper: leaves share a single y-coordinate and internal
        nodes line up at the height of their deepest descendant.
        """
        self = cls(level_step=level_step)
        counter = [0]

        def _layout(n: PhyloTree) -> Tuple[float, float]:
            if n.is_leaf():
                x = float(counter[0])
                counter[0] += 1
                self.positions[id(n)] = (x, 0.0)
                self.leaf_ids.add(id(n))
                return x, 0.0
            assert n.left is not None and n.right is not None
            lx, ly = _layout(n.left)
            rx, ry = _layout(n.right)
            x = (lx + rx) / 2.0
            y = max(ly, ry) + self.level_step
            self.positions[id(n)] = (x, y)
            return x, y

        _root_x, _root_y = _layout(tree)
        self.max_depth = _root_y
        self.n_leaves = tree.n_leaves()
        return self

    def edge_segment(
        self, parent: PhyloTree, child: PhyloTree
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return the ``((px, py), (cx, cy))`` segment for an edge."""
        return self.positions[id(parent)], self.positions[id(child)]


__all__ = ["TreeLayout"]
