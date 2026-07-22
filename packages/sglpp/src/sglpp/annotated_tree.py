"""Annotated phylogenetic tree — annotation data model.

Typed primitives for styling and placing annotations on a
:class:`~sglpp.tree.PhyloTree`, plus the :class:`AnnotatedTree`
annotation store.  The matplotlib renderer lives in
:mod:`sglpp.tree_renderer` (behind the ``[viz]`` extra) so this module
stays free of the visualization dependency.

Layers:

1. :class:`RichText` — styled text (color, italic, bold, family, size).
2. :class:`EdgeLabel` / :class:`EdgeMarker` — annotations placed on a branch.
3. :class:`NodeAnnotation` / :class:`AnnotatedTree` — per-node and per-edge
   annotation store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from sglpp.tree import PhyloTree



@dataclass(frozen=True)
class RichText:
    """A piece of styled text.

    Inline lists of :class:`RichText` build multi-color labels — useful for
    internal-node gene strings where each character may be tinted by its
    edge event (loss = blue, gain = red).
    """

    text: str
    color: str = "#1a1a1a"
    italic: bool = False
    bold: bool = False
    family: str = "serif"
    size: float = 12.0

    @staticmethod
    def join(parts: Sequence["RichText"]) -> str:
        """Concatenate a sequence of RichText into one plain string."""
        return "".join(p.text for p in parts)


@dataclass(frozen=True)
class EdgeLabel:
    """Text annotation placed on an edge (parent → child).

    *position* ranges over the branch (0 = at parent, 1 = at child); the
    ortho-L layout makes the midpoint a natural anchor for event labels.
    *offset* pushes the label perpendicular to the edge direction in data
    units (positive = above the branch in the paper convention).  *side*
    controls horizontal alignment (``-1`` = left, ``0`` = centered,
    ``+1`` = right).
    """

    text: RichText
    position: float = 0.5
    offset: float = 0.0
    side: int = 0


@dataclass(frozen=True)
class EdgeMarker:
    """Symbol placed on an edge (parent → child).

    Supported symbols: ``"x"`` (matplotlib ``x`` marker, e.g. losses) and
    ``"o"`` (filled circle, e.g. gains).
    """

    symbol: str
    color: str = "#1a1a1a"
    position: float = 0.5
    size: float = 8.0




@dataclass
class NodeAnnotation:
    """Per-node annotation bundle."""

    label: Optional[Union[RichText, List[RichText]]] = None
    marker: Optional[str] = None
    marker_color: str = "#1a1a1a"
    marker_size: float = 8.0
    label_offset: Tuple[float, float] = (0.15, 0.0)


EdgeAnnotation = Union[EdgeLabel, EdgeMarker]


@dataclass
class AnnotatedTree:
    """PhyloTree plus per-node and per-edge annotations."""

    tree: PhyloTree
    node_annotations: Dict[int, NodeAnnotation] = field(default_factory=dict)
    edge_annotations: Dict[Tuple[int, int], List[EdgeAnnotation]] = field(
        default_factory=dict
    )

    def annotate_node(self, node: PhyloTree, **fields: Any) -> "AnnotatedTree":
        """Merge fields into the annotation of *node*."""
        current = self.node_annotations.get(id(node), NodeAnnotation())
        for key, value in fields.items():
            setattr(current, key, value)
        self.node_annotations[id(node)] = current
        return self

    def annotate_edge(
        self,
        parent: PhyloTree,
        child: PhyloTree,
        annotation: EdgeAnnotation,
    ) -> "AnnotatedTree":
        """Append an EdgeLabel or EdgeMarker on the *parent → child* edge."""
        key = (id(parent), id(child))
        self.edge_annotations.setdefault(key, []).append(annotation)
        return self


__all__ = [
    "RichText",
    "EdgeLabel",
    "EdgeMarker",
    "NodeAnnotation",
    "AnnotatedTree",
]
