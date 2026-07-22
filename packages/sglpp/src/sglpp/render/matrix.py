"""Matrix drawing primitives — single entry point for the SGLPP heatmap.

Used by :mod:`sglpp.render.tree` (mingap figure) and the standalone
``scripts/render.py`` CLI.  Rows that can never contribute cost (fewer
than two ``1`` s) are dropped — matches the pre-filter
:func:`mingap.solve` applies internally.
"""

from __future__ import annotations

from sglpp.render.palette import PALETTE_RGB
from sglpp.scenario import GENE_NAMES


def draw_matrix(ax, res, perm, labels: dict[int, str] | None = None) -> None:
    """Draw the SGLPP matrix (columns reordered by *perm*) onto *ax*.

    Trivial rows (``count(1) < 2``) are dropped — they always contribute
    cost 0, matching the pre-filter :func:`mingap.solve` applies
    internally.  When every row is trivial a single text annotation is
    rendered instead of an empty axes.  *labels* maps edge ids to custom
    row prefixes (e.g. ``{3: "G1", 4: "L1"}``); defaults to ``e{id}``.
    """
    m = res.build_matrix()
    keep = [i for i, row in enumerate(m) if row.count(1) >= 2]
    m2 = [[m[i][c] for c in perm] for i in keep]
    n_rows = len(m2)
    n_cols = len(perm)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [GENE_NAMES[c] if c < len(GENE_NAMES) else str(c) for c in perm],
        fontsize=10,
    )
    ax.xaxis.tick_top()

    if n_rows == 0:
        ax.text(
            0.5,
            0.5,
            "(no non-trivial rows)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="#888",
        )
        ax.set_yticks([])
        return

    img = [[PALETTE_RGB[v] for v in row] for row in m2]
    ax.imshow(img, aspect="auto")

    edge_labels = [
        f"{(labels.get(e.id, f'e{e.id}') if labels else f'e{e.id}')} {kind}"
        for e in res.edges
        for kind in ("loss", "gain")
    ]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([edge_labels[i] for i in keep], fontsize=7, family="monospace")

    for i, row in enumerate(m2):
        for j, v in enumerate(row):
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if v == 0 else "#1a1a1a",
            )


__all__ = ["draw_matrix"]
