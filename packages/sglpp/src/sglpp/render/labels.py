"""Label helpers — text/formatting conversions for gene sets and orders.

Single source of truth for rendering frozensets / permutations as
human-readable strings.
"""

from __future__ import annotations

from sglpp.scenario import GENE_NAMES


def gene_label(
    gs,
    gene_names: str = GENE_NAMES,
    *,
    order=None,
) -> str:
    """Space-separated gene list, e.g. ``"a b c"``. Empty → ``"∅"``.

    If *order* is given, genes are emitted in that order (intersected with
    *gs*); otherwise genes are sorted ascending.
    """
    if not gs:
        return "∅"
    items = order if order is not None else sorted(gs)
    return " ".join(gene_names[g] for g in items if g in gs and g < len(gene_names))


def setstr(gs, gene_names: str = GENE_NAMES) -> str:
    """Comma-separated set braces, e.g. ``"{a,b,c}"``. Empty → ``"∅"``."""
    if not gs:
        return "∅"
    return (
        "{" + ",".join(gene_names[g] for g in sorted(gs) if g < len(gene_names)) + "}"
    )


def order_str(
    order,
    signed: bool = False,
    gene_names: str = GENE_NAMES,
) -> str:
    """Gene order as a compact string. Signed: lowercase = +, UPPER = −."""
    if signed:
        return "".join(
            gene_names[abs(g)].upper() if g < 0 else gene_names[g] for g in order
        )
    return "".join(gene_names[g] for g in order)


__all__ = ["gene_label", "order_str", "setstr"]