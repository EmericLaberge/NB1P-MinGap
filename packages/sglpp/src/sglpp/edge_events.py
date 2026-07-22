"""Edge event markers (× loss, • gain, ↺ inversion) shared by every tree renderer.

Single source of truth for the event-marker style used by the Streamlit Solve
and Inversions pages *and* the standalone ``scripts/render`` tool.
Needs matplotlib (the ``viz`` extra); every renderer already pulls it in.
"""

from __future__ import annotations

from sglpp.render import GAIN_COLOR as _GAIN, INV_COLOR as _INV, LOSS_COLOR as _LOSS


def draw_edge_events(
    ax,
    parent_xy: tuple[float, float],
    child_xy: tuple[float, float],
    losses: frozenset[int],
    gains: frozenset[int],
    setlabel,
    *,
    font: str,
    loss_n: int,
    gain_n: int,
    inv: int = 0,
) -> tuple[int, int]:
    """Draw loss (×) and gain (•) event markers on the edge parent → child.

    Returns the updated ``(loss_n, gain_n)`` running counters.  *setlabel*
    renders a gene-set as a ``{a,b,c}`` string; *font* is the matplotlib
    family.  *inv* (>0, Inversions page only) adds a violet ``↺N`` count
    above the midpoint.
    """
    (px, py), (cx, cy) = parent_xy, child_xy
    if not losses and not gains and not inv:
        return loss_n, gain_n
    side = -1 if cx < px else 1
    if losses:
        loss_n += 1
    if gains:
        gain_n += 1
    lx = ly = gx = gy = None
    if losses and gains:
        lx, ly = px + (cx - px) * 0.30, py + (cy - py) * 0.30
        gx, gy = px + (cx - px) * 0.70, py + (cy - py) * 0.70
    elif losses:
        lx, ly = px + (cx - px) * 0.50, py + (cy - py) * 0.50
    elif gains:                              # ne pas dessiner un gain vide
        gx, gy = px + (cx - px) * 0.50, py + (cy - py) * 0.50
    if lx is not None:
        ax.text(
            lx,
            ly,
            "×",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=_LOSS,
            family=font,
            zorder=5,
        )
        ax.text(
            lx + side * 0.12,
            ly,
            f"L{loss_n} {setlabel(losses)}",
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=7.5,
            style="italic",
            color=_LOSS,
            family=font,
            zorder=5,
        )
    if gx is not None:
        ax.plot(
            gx,
            gy,
            marker="o",
            markersize=5,
            color=_GAIN,
            markeredgecolor=_GAIN,
            zorder=5,
        )
        ax.text(
            gx + side * 0.12,
            gy,
            f"G{gain_n} {setlabel(gains)}",
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=7.5,
            style="italic",
            color=_GAIN,
            family=font,
            zorder=5,
        )
    if inv:
        ax.text(
            (px + cx) / 2.0,
            (py + cy) / 2.0 + 0.12,
            f"↺{inv}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=_INV,
            family=font,
            zorder=5,
        )
    return loss_n, gain_n


__all__ = ["draw_edge_events"]
