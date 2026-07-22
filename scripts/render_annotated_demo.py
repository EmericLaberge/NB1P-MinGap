#!/usr/bin/env python3
"""``render_annotated_demo`` — demonstrate the annotated-tree renderer.

Builds a small synthetic SGLPP scenario, annotates every node and edge with
multi-color gene content, loss (×) markers, gain (●) markers, and italic
event labels, then writes a PNG/PDF/SVG figure.

    uv run python scripts/render_annotated_demo.py [out.png]

The default output is ``/tmp/annotated_demo.png``.
"""

from __future__ import annotations

import sys

from sglpp import (  # noqa: E402
    AnnotatedTree,
    EdgeLabel,
    EdgeMarker,
    PhyloTree,
    RichText,
    TreeRenderer,
    parse_tree,
)

GENE_NAMES = "abcdefghijklmnopqrstuvwxyz"


def _build_tree() -> PhyloTree:
    """((a,b),(c,d)) — symmetric 4-leaf tree, easiest to read."""
    return parse_tree(((0, 1), (2, 3)))


def _annotate(annotated: AnnotatedTree) -> None:
    """Decorate the demo tree with a mix of node and edge annotations."""
    tree = annotated.tree

    # leaf names below each leaf
    for leaf in (tree.left.left, tree.left.right, tree.right.left, tree.right.right):
        annotated.annotate_node(leaf, label=RichText(GENE_NAMES[leaf.label], size=14.0))

    # internal-node gene content (italic gray)
    annotated.annotate_node(
        tree.left,
        label=RichText("ab", color="#9a9a9a", italic=True, size=14.0),
    )
    annotated.annotate_node(
        tree,
        label=RichText("abcd", color="#9a9a9a", italic=True, size=14.0),
    )

    # left subtree: a single loss event on the edge to leaf 0
    annotated.annotate_edge(
        tree.left,
        tree.left.left,
        EdgeMarker(symbol="x", color="#1a1a1a", position=0.5),
    )
    annotated.annotate_edge(
        tree.left,
        tree.left.left,
        EdgeLabel(
            RichText("L1 {a}", color="#dc2626", italic=True, size=9.0),
            position=0.5,
            side=1,
        ),
    )

    # right subtree: a single gain event on the edge to leaf 3
    annotated.annotate_edge(
        tree.right,
        tree.right.right,
        EdgeMarker(symbol="o", color="#16a34a", position=0.5, size=8.0),
    )
    annotated.annotate_edge(
        tree.right,
        tree.right.right,
        EdgeLabel(
            RichText("G1 {d}", color="#16a34a", italic=True, size=9.0),
            position=0.5,
            side=1,
        ),
    )


    annotated.annotate_node(
        tree,
        marker="o",
        marker_color="#1a1a1a",
        marker_size=8.0,
    )


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/annotated_demo.png"
    annotated = AnnotatedTree(_build_tree())
    _annotate(annotated)
    TreeRenderer(annotated).render(out)
    print(f"rendered {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
