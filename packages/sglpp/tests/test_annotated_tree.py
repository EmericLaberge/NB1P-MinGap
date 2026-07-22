"""Smoke tests for the annotated-tree renderer."""

from __future__ import annotations

from pathlib import Path

import pytest
from sglpp.annotated_tree import AnnotatedTree, EdgeLabel, EdgeMarker, RichText
from sglpp.tree_renderer import TreeRenderer
from sglpp.tree import PhyloTree, parse_tree


def _four_leaf_tree() -> PhyloTree:
    """((0,1),(2,3)) — two internal nodes, four leaves."""
    return parse_tree(((0, 1), (2, 3)))


def _leaf_name(label: int) -> str:
    return "abcd"[label]


def test_rich_text_join() -> None:
    a = RichText("ab", color="#1a1a1a")
    b = RichText("c", color="#c0392b", italic=True)
    assert RichText.join([a, b]) == "abc"


def test_annotate_node_and_edge(tmp_path: Path) -> None:
    tree = _four_leaf_tree()
    annotated = AnnotatedTree(tree)

    leaves = (tree.left.left, tree.left.right, tree.right.left, tree.right.right)
    leaf_a = leaves[0]
    for leaf in leaves:
        annotated.annotate_node(leaf, label=RichText(_leaf_name(leaf.label), size=14.0))

    annotated.annotate_node(tree, label=RichText("abcd", italic=True, color="#9a9a9a"))
    annotated.annotate_edge(tree.left, leaf_a, EdgeMarker(symbol="x", color="#1a1a1a"))
    annotated.annotate_edge(
        tree.left,
        leaf_a,
        EdgeLabel(
            RichText(
                f"L1 {{{_leaf_name(leaf_a.label)}}}",
                color="#dc2626",
                italic=True,
                size=9.0,
            ),
            position=0.5,
            side=1,
        ),
    )

    out = tmp_path / "tree.png"
    renderer = TreeRenderer(annotated)
    assert renderer.render(str(out)) == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_layout_returns_positions() -> None:
    tree = _four_leaf_tree()
    annotated = AnnotatedTree(tree)
    renderer = TreeRenderer(annotated)
    renderer.layout()
    assert len(renderer._positions) == tree.n_leaves() + 3
    for x, y in renderer._positions.values():
        assert isinstance(x, float)
        assert isinstance(y, float)


def test_multicolor_node_label(tmp_path: Path) -> None:
    """A node label with two RichText segments exercises the overlay path."""
    tree = _four_leaf_tree()
    annotated = AnnotatedTree(tree)
    annotated.annotate_node(
        tree,
        label=[
            RichText("ab", color="#9a9a9a", italic=True, size=14.0),
            RichText("c", color="#c0392b", italic=True, size=14.0),
        ],
    )
    out = tmp_path / "multicolor.png"
    TreeRenderer(annotated).render(str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_without_layout(tmp_path: Path) -> None:
    """render() must call layout() automatically when no positions exist."""
    tree = _four_leaf_tree()
    annotated = AnnotatedTree(tree)
    out = tmp_path / "auto_layout.png"
    TreeRenderer(annotated).render(str(out))
    assert out.exists()


def test_unknown_marker_raises() -> None:
    tree = _four_leaf_tree()
    annotated = AnnotatedTree(tree)
    annotated.annotate_edge(
        tree.left,
        tree.left.left,
        EdgeMarker(symbol="?", color="#1a1a1a"),
    )
    with pytest.raises(ValueError, match="unsupported marker symbol"):
        TreeRenderer(annotated).render("/tmp/should_fail.png")


def test_export_pdf_and_svg(tmp_path: Path) -> None:
    """The same pipeline supports PDF/SVG output — format inferred from suffix."""
    tree = _four_leaf_tree()
    annotated = AnnotatedTree(tree)
    annotated.annotate_node(tree, label=RichText("abcd", italic=True, size=14.0))
    for path in (tmp_path / "out.pdf", tmp_path / "out.svg"):
        TreeRenderer(annotated).render(str(path))
        assert path.exists()
        assert path.stat().st_size > 0
