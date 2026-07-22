"""Poster-style counterexample visualization (matplotlib → PNG).
Requires the viz extra (pip install mininv[viz]).

The layout uses :class:`~sglpp.layout.TreeLayout` (the canonical
cladogram coordinates) and the permutation search comes from
:mod:`sglpp.search.perm`; this module is now strictly drawing.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sglpp.display import scenario_loss_labels
from sglpp.layout import TreeLayout
from sglpp.render import PALETTE_POSTER
from sglpp.scenario import Scenario
from sglpp.search.perm import best_partial_perm, violations_of
from sglpp.tree import PhyloTree

# Cell colours for the ternary semantics (0=present, 1=lost, 2=ancestor).
# Source of truth: :data:`sglpp.render.PALETTE_POSTER`.
PALETTE = PALETTE_POSTER


def visualize(
    scenario: Scenario,
    matrix: list[list[int]],
    idx: int,
    outdir: str = ".",
    show_perm: bool = True,
):
    """Render the counterexample tree + matrix as a single PNG."""
    tree = scenario.tree
    n_genes = scenario.n_genes
    n_leaves = tree.n_leaves()

    # Canonical layout, y-flipped so the root sits at the top (standard
    # phylogenetic convention: root y=0, leaves y=-max_depth).
    layout = TreeLayout.build(tree, level_step=1.0)
    positions = {nid: (x, -y) for nid, (x, y) in layout.positions.items()}
    max_depth = layout.max_depth

    x_pad = 0.5
    xmin_plot = 0
    xmax_plot = (n_leaves - 1) + 2 * x_pad

    fig_w = max(14, 4 + 2 * n_leaves + 1.5 * n_genes)
    fig_h = 7
    fig, (ax_t, ax_m) = plt.subplots(
        1,
        2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1.3, 1.4]},
    )
    fig.suptitle(
        f"Counter-example #{idx} — NB1P fails ({n_genes} genes, {n_leaves} leaves)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    for ei in scenario.edges:
        parent = ei.parent
        child = ei.child
        px, py = positions[id(parent)]
        cx, cy = positions[id(child)]
        ax_t.plot([px, cx], [py, cy], "k-", linewidth=2, zorder=1)

    loss_edges = [
        ei for ei in scenario.edges if scenario.losses.get(ei.id, frozenset())
    ]
    for ei in loss_edges:
        parent = ei.parent
        child = ei.child
        px, py = positions[id(parent)]
        cx, cy = positions[id(child)]
        loss = scenario.losses.get(ei.id, frozenset())
        gene_str = ",".join(scenario.gene_name(g) for g in sorted(loss))
        is_left = cx < px
        xoff = -0.15 if is_left else 0.15
        lx = (px + cx) / 2 + xoff
        ly = (py + cy) / 2
        ax_t.annotate(
            f"$L_{{{ei.id}}}$:{{{gene_str}}}",
            (lx, ly),
            fontsize=9,
            color="red",
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#FFFACD", ec="gray", lw=0.5),
            zorder=5,
        )

    def _draw_nodes(n: PhyloTree):
        assert n.left is not None and n.right is not None
        x, y = positions[id(n)]
        if not n.is_leaf():
            ax_t.plot(
                x,
                y,
                "o",
                markersize=6,
                color="white",
                markeredgecolor="black",
                markeredgewidth=1,
                zorder=2,
            )
            _draw_nodes(n.left)
            _draw_nodes(n.right)

    genomes = scenario.leaf_genomes()
    for ei in scenario.edges:
        if ei.child.is_leaf():
            x, y = positions[id(ei.child)]
            genome = genomes.get(ei.child.label, frozenset())
            txt = ",".join(scenario.gene_name(g) for g in sorted(genome))
            txt = txt or "∅"
            ax_t.plot(
                x,
                y,
                "o",
                markersize=26,
                color="#B0D4F1",
                markeredgecolor="#2C5F8D",
                markeredgewidth=1.5,
                zorder=3,
            )
            ax_t.text(
                x,
                y,
                txt,
                fontsize=9,
                ha="center",
                va="center",
                zorder=4,
                color="#1A3A52",
                fontweight="bold",
            )

    rx, ry = positions[id(tree)]
    ax_t.plot(rx, ry + 0.25, "^", markersize=16, color="#2C5F8D", zorder=3)
    root_genes = ",".join(scenario.gene_name(g) for g in range(n_genes))
    ax_t.text(
        rx,
        ry + 0.50,
        root_genes,
        fontsize=11,
        ha="center",
        va="bottom",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="#E8F0F8", ec="#2C5F8D", lw=0.8),
    )

    ax_t.set_xlim(xmin_plot - 0.4, xmax_plot + 0.4)
    ax_t.set_ylim(-max_depth - 0.8, 0.8)
    ax_t.set_aspect("auto")
    ax_t.axis("off")
    ax_t.set_title("Phylogenetic tree", fontsize=12)

    if matrix:
        data = np.array(matrix)
        cmap = ListedColormap([PALETTE[0], PALETTE[1], PALETTE[2]])
        ax_m.imshow(data, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        labels = scenario_loss_labels(scenario)
        ax_m.set_yticks(range(len(matrix)))
        ax_m.set_yticklabels(labels, fontsize=9)
        ax_m.set_xticks(range(n_genes))
        ax_m.set_xticklabels(
            [scenario.gene_name(g) for g in range(n_genes)],
            fontsize=13,
            fontweight="bold",
        )
        for i in range(len(matrix)):
            for j in range(n_genes):
                v = matrix[i][j]
                ax_m.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white" if v == 1 else "black",
                )
        ax_m.set_title(
            "Ternary matrix  (0=present  1=lost  2=ancestor)",
            fontsize=11,
        )

        ax_m.set_xticks(np.arange(-0.5, n_genes, 1), minor=True)
        ax_m.set_yticks(np.arange(-0.5, len(matrix), 1), minor=True)
        ax_m.grid(which="minor", color="white", linewidth=2)
        ax_m.tick_params(which="minor", length=0)

        if show_perm and n_genes <= 8:
            best = best_partial_perm(matrix, n_genes)
            if best is not None:
                violated_rows = violations_of(best, matrix, n_genes)
                perm_str = ",".join(scenario.gene_name(g) for g in best)
                n_viol = len(violated_rows)
                if n_viol == 0:
                    msg = f"Best permutation π = ({perm_str})  →  NB1P satisfied"
                    color = "green"
                else:
                    row_labels = scenario_loss_labels(scenario)
                    msg = (
                        f"Best permutation π = ({perm_str})  →  "
                        f"{n_viol} row(s) violated: "
                        f"{[row_labels[i] for i in violated_rows]}"
                    )
                    color = "red"
                ax_m.text(
                    0.5,
                    -0.18,
                    msg,
                    transform=ax_m.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FFFACD", ec=color, lw=0.8),
                )
    else:
        ax_m.text(
            0.5,
            0.5,
            "Empty matrix",
            transform=ax_m.transAxes,
            ha="center",
        )

    legend_patches = [
        mpatches.Patch(color=PALETTE[0], label="0 present"),
        mpatches.Patch(color=PALETTE[1], label="1 lost here"),
        mpatches.Patch(color=PALETTE[2], label="2 lost by ancestor"),
    ]
    ax_m.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fname = f"{outdir}/counterexample_{idx}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {fname}")
    return fname


__all__ = ["PALETTE", "visualize"]
