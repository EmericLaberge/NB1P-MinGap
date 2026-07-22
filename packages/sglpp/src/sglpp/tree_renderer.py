"""Matplotlib renderer for :class:`~sglpp.annotated_tree.AnnotatedTree`.

Extracted from ``annotated_tree`` so the annotation *data model* stays
matplotlib-free; only this module pays the ``[viz]`` import cost.  Branches
are drawn as ortho-cladogram L-shapes (vertical drop then horizontal segment)
to keep the paper aesthetic.  Layout reuses
:class:`sglpp.layout.TreeLayout`, so figures are interchangeable
across the app, scripts, and any future rendering tool.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sglpp.annotated_tree import (  # noqa: E402
    AnnotatedTree,
    EdgeLabel,
    EdgeMarker,
    RichText,
)
from sglpp.layout import TreeLayout  # noqa: E402
from sglpp.tree import PhyloTree  # noqa: E402

__all__ = ["TreeRenderer"]


class TreeRenderer:
    """Layout (:class:`TreeLayout`) + render (matplotlib) for an AnnotatedTree.

    Branches are drawn as L-shapes so the figure stays in the ortho-cladogram
    style of the paper: vertical drop from the parent, then horizontal
    segment to the child.  ``TreeLayout`` is the same recursive cladogram
    layout used by the Streamlit pages, so positions are interchangeable
    across the app, scripts, and any future rendering tool.  Matplotlib
    handles every glyph so multi-color text, italic styling, and per-event
    markers stay full-fidelity.
    """

    def __init__(
        self,
        annotated: AnnotatedTree,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 170,
        background: str = "white",
        edge_color: str = "#7d7d7d",
        edge_width: float = 2.4,
        level_step: float = 0.6,
    ) -> None:
        self.ann = annotated
        self.figsize = figsize
        self.dpi = dpi
        self.background = background
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.level_step = level_step
        self._positions: Dict[int, Tuple[float, float]] = {}

    # ── layout (TreeLayout — same as the Streamlit app) ───────────────

    def layout(self, level_step: Optional[float] = None) -> "TreeRenderer":
        """Compute node positions with :class:`TreeLayout`."""
        step = self.level_step if level_step is None else level_step
        tree_layout = TreeLayout.build(self.ann.tree, level_step=step)
        self._positions = dict(tree_layout.positions)
        return self

    # ── render (matplotlib) ────────────────────────────────────────────

    def render(self, out_path: str) -> str:
        """Layout (if needed) then write a PNG/PDF/SVG to *out_path*."""
        if not self._positions:
            self.layout()

        tree = self.ann.tree
        all_nodes: List[PhyloTree] = []
        self._collect(tree, all_nodes)

        figure_width = (
            self.figsize[0] if self.figsize else max(8.0, tree.n_leaves() * 1.6)
        )
        figure_height = self.figsize[1] if self.figsize else 5.5
        figure, axis = plt.subplots(
            figsize=(figure_width, figure_height),
            dpi=self.dpi,
            facecolor=self.background,
        )
        axis.set_facecolor(self.background)
        axis.set_aspect("auto")
        axis.axis("off")

        self._set_limits(axis)
        self._axis = axis
        self._figure = figure

        # branches first, so node markers and labels sit on top
        for node in all_nodes:
            if node.is_leaf():
                continue
            assert node.left is not None and node.right is not None
            self._draw_branch(node, node.left)
            self._draw_branch(node, node.right)
        for node in all_nodes:
            self._draw_node(node)

        figure.savefig(
            out_path,
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor=self.background,
        )
        plt.close(figure)
        return out_path

    # ── internals ───────────────────────────────────────────────────────

    def _collect(self, node: PhyloTree, out: List[PhyloTree]) -> None:
        out.append(node)
        if not node.is_leaf():
            assert node.left is not None and node.right is not None
            self._collect(node.left, out)
            self._collect(node.right, out)

    def _set_limits(self, axis: Any) -> None:
        xs = [point[0] for point in self._positions.values()]
        ys = [point[1] for point in self._positions.values()]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        pad = max(1.0, (xmax - xmin) * 0.05)
        axis.set_xlim(xmin - pad, xmax + pad + 1.0)
        axis.set_ylim(ymin - pad, ymax + pad + 0.8)

    def _draw_branch(self, parent: PhyloTree, child: PhyloTree) -> None:
        axis = self._axis
        parent_xy = self._positions[id(parent)]
        child_xy = self._positions[id(child)]
        px, py = parent_xy
        cx, cy = child_xy
        if px == cx:
            axis.plot(
                [px, cx],
                [py, cy],
                color=self.edge_color,
                lw=self.edge_width,
                solid_capstyle="round",
                zorder=1,
            )
        else:
            axis.plot(
                [px, px],
                [py, cy],
                color=self.edge_color,
                lw=self.edge_width,
                solid_capstyle="round",
                zorder=1,
            )
            axis.plot(
                [px, cx],
                [cy, cy],
                color=self.edge_color,
                lw=self.edge_width,
                solid_capstyle="round",
                zorder=1,
            )
        for annotation in self.ann.edge_annotations.get((id(parent), id(child)), []):
            if isinstance(annotation, EdgeLabel):
                self._draw_edge_label(parent, child, annotation)
            elif isinstance(annotation, EdgeMarker):
                self._draw_edge_marker(parent, child, annotation)

    def _interpolated(self, parent_xy, child_xy, position: float):
        (px, py), (cx, cy) = parent_xy, child_xy
        return px + (cx - px) * position, py + (cy - py) * position

    def _draw_edge_label(
        self,
        parent: PhyloTree,
        child: PhyloTree,
        annotation: EdgeLabel,
    ) -> None:
        axis = self._axis
        parent_xy = self._positions[id(parent)]
        child_xy = self._positions[id(child)]
        x, y = self._interpolated(parent_xy, child_xy, annotation.position)
        x += 0.15 if annotation.side > 0 else (-0.15 if annotation.side < 0 else 0.0)
        y += annotation.offset
        horizontal_alignment = (
            "left"
            if annotation.side > 0
            else "right"
            if annotation.side < 0
            else "center"
        )
        rich = annotation.text
        axis.text(
            x,
            y,
            rich.text,
            ha=horizontal_alignment,
            va="center",
            fontsize=rich.size,
            family=rich.family,
            fontstyle="italic" if rich.italic else "normal",
            fontweight="bold" if rich.bold else "normal",
            color=rich.color,
            zorder=5,
        )

    def _draw_edge_marker(
        self,
        parent: PhyloTree,
        child: PhyloTree,
        annotation: EdgeMarker,
    ) -> None:
        parent_xy = self._positions[id(parent)]
        child_xy = self._positions[id(child)]
        x, y = self._interpolated(parent_xy, child_xy, annotation.position)
        if annotation.symbol == "x":
            self._axis.plot(
                x,
                y,
                marker="x",
                color=annotation.color,
                markersize=annotation.size,
                markeredgewidth=2.0,
                zorder=4,
            )
        elif annotation.symbol == "o":
            self._axis.plot(
                x,
                y,
                marker="o",
                color=annotation.color,
                markersize=annotation.size,
                zorder=4,
            )
        else:
            raise ValueError(f"unsupported marker symbol: {annotation.symbol!r}")

    def _draw_node(self, node: PhyloTree) -> None:
        annotation = self.ann.node_annotations.get(id(node))
        if annotation is None:
            return
        axis = self._axis
        x, y = self._positions[id(node)]
        if annotation.marker:
            axis.plot(
                x,
                y,
                marker=annotation.marker,
                color=annotation.marker_color,
                markersize=annotation.marker_size,
                zorder=3,
            )
        if annotation.label is None:
            return
        labels = (
            annotation.label
            if isinstance(annotation.label, list)
            else [annotation.label]
        )
        ox, oy = annotation.label_offset
        full_text = RichText.join(labels)
        first = labels[0]
        axis.text(
            x + ox,
            y + oy,
            full_text,
            ha="left",
            va="center",
            fontsize=first.size,
            family=first.family,
            fontstyle="italic" if first.italic else "normal",
            fontweight="bold" if first.bold else "normal",
            color=first.color,
            zorder=4,
        )
        # overlay characters whose color differs from the first segment
        if len(labels) == 1 and labels[0].color == first.color:
            return
        self._figure.canvas.draw()  # type: ignore[union-attr]
        last = axis.texts[-1]
        bounding_box = last.get_window_extent()
        inverse = axis.transData.inverted()
        x0, _ = inverse.transform((bounding_box.x0, bounding_box.y0))
        x1, _ = inverse.transform((bounding_box.x1, bounding_box.y1))
        char_width = (x1 - x0) / max(len(full_text), 1)
        cursor = 0
        for segment in labels:
            for _char in segment.text:
                if segment.color != first.color:
                    cx = x0 + char_width * (cursor + 0.5)
                    axis.text(
                        cx,
                        y + oy,
                        _char,
                        ha="center",
                        va="center",
                        fontsize=segment.size,
                        family=segment.family,
                        fontstyle="italic" if segment.italic else "normal",
                        fontweight="bold" if segment.bold else "normal",
                        color=segment.color,
                        zorder=5,
                    )
                cursor += 1
