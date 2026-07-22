"""Tree drawing primitives — single source of truth for the SGLPP/MinGap figures.

Public entry point:

- :func:`render_mingap` — SGLPP tree + MinGap-optimal matrix side by side

Shares :class:`~sglpp.layout.TreeLayout` +
:func:`~sglpp.edge_events.draw_edge_events` + palette + label helpers, so a new
caller only needs to import from this module.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from sglpp.edge_events import draw_edge_events
from sglpp.layout import TreeLayout
from sglpp.render.labels import gene_label, setstr
from sglpp.render.matrix import draw_matrix
from sglpp.render.palette import (
    ANN_COLOR,
    EDGE_COLOR,
    FONT_SERIF,
    TEXT_COLOR,
)
from sglpp.scenario import GENE_NAMES


def _draw_sglpp_tree(ax, tree, res, leaf_orders) -> None:
    """Draw the SGLPP tree (genomes + ×/● markers) onto *ax*."""
    edges = res.edges
    nc = res.node_contents
    layout = TreeLayout.build(tree, level_step=0.45)
    pos = layout.positions
    leaf_ids = layout.leaf_ids
    max_depth = layout.max_depth
    n_leaves = layout.n_leaves

    slot_w = 1.1
    xmax = (n_leaves - 1) * slot_w
    pad_x = 1.2
    pad_y_top = 0.7
    pad_y_bot = 0.6

    for e in edges:
        px, py = pos[id(e.parent)]
        cx, cy = pos[id(e.child)]
        ax.plot(
            [px, cx],
            [py, cy],
            color=EDGE_COLOR,
            linewidth=1.2,
            solid_capstyle="round",
            zorder=1,
        )

    LEVEL_STEP = layout.level_step
    rx0, ry0 = pos[id(tree)]
    fx, fy = rx0, ry0 + LEVEL_STEP / 1.5
    ax.plot(
        [fx, rx0],
        [fy, ry0],
        color=EDGE_COLOR,
        linewidth=1.2,
        solid_capstyle="round",
        zorder=1,
    )
    ax.plot(
        fx,
        fy,
        marker="o",
        markersize=5,
        color=TEXT_COLOR,
        markeredgecolor=TEXT_COLOR,
        zorder=5,
    )
    ax.text(
        fx + 0.08,
        fy,
        setstr(nc[id(tree)]),
        ha="left",
        va="center",
        fontsize=7.5,
        style="italic",
        color=ANN_COLOR,
        family=FONT_SERIF,
        zorder=5,
    )

    ax.text(
        rx0 + 0.18,
        ry0 + 0.02,
        gene_label(nc[id(tree)]),
        ha="center",
        va="bottom",
        fontsize=8,
        style="italic",
        fontweight="bold",
        color=TEXT_COLOR,
        family=FONT_SERIF,
        zorder=4,
    )

    for e in edges:
        nid = id(e.child)
        x, y = pos[nid]
        if nid in leaf_ids:
            ax.text(
                x,
                y - 0.08,
                gene_label(nc.get(nid, frozenset()), order=leaf_orders.get(nid)),
                ha="center",
                va="top",
                fontsize=9,
                style="italic",
                color=TEXT_COLOR,
                family=FONT_SERIF,
                zorder=4,
            )
        else:
            nudge = -0.15 if e.child is e.parent.left else 0.15
            ax.text(
                x + nudge,
                y + 0.02,
                gene_label(nc.get(nid, frozenset())),
                ha="center",
                va="bottom",
                fontsize=9,
                style="italic",
                color=TEXT_COLOR,
                family=FONT_SERIF,
                zorder=4,
            )

    lc = gc = 0
    for e in edges:
        lc, gc = draw_edge_events(
            ax,
            pos[id(e.parent)],
            pos[id(e.child)],
            res.losses.get(e.id, frozenset()),
            res.gains.get(e.id, frozenset()),
            setstr,
            font=FONT_SERIF,
            loss_n=lc,
            gain_n=gc,
        )

    ax.set_xlim(-pad_x, xmax + pad_x)
    ax.set_ylim(-pad_y_bot, max_depth + pad_y_top)
    ax.set_aspect("auto")
    ax.axis("off")


def render_mingap(tree, res, leaf_orders, perm, cost, out: str, labels: dict[int, str] | None = None) -> None:
    """Combined SGLPP tree (left) + MinGap-optimal matrix (right), one PNG."""
    layout = TreeLayout.build(tree)
    n_leaves = layout.n_leaves
    n_genes = res.n_genes
    fig_w = max(12.0, 3.0 + n_leaves * 1.0 + n_genes * 0.7)
    fig_h = max(5.0, layout.max_depth + 1.8)
    fig, (ax_t, ax_m) = plt.subplots(
        1,
        2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )
    _draw_sglpp_tree(ax_t, tree, res, leaf_orders)
    draw_matrix(ax_m, res, perm, labels)
    ax_m.set_title(
        f"MinGap = {cost}  |  order {' '.join(GENE_NAMES[g] for g in perm)}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


__all__ = ["render_mingap"]