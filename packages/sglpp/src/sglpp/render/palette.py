"""Single source of truth for every render-side color and style constant.

Two distinct palettes exist by design:

- :data:`PALETTE_HEX` / :data:`PALETTE_RGB`: bold red/teal/gray for the
  ternary matrix semantics.
- :data:`PALETTE_POSTER`: pastel LaTeX-matching palette
  (``red!15``, ``teal!20``, ``gray!15``) used by the paper-style
  ``viz.visualize`` poster.

Tree text/edge/ann colors and event-marker colors live next to the palette
so a renderer only needs to import :mod:`sglpp.render.palette`.
"""

from __future__ import annotations


#: Cell colours for the ternary matrix semantics (0=present, 1=lost, 2=ancestral).
PALETTE_HEX: dict[int, str] = {
    0: "#ef4444",
    1: "#5eead4",
    2: "#b0b8c4",
}


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    """Convert ``"#rrggbb"`` to a normalised ``(r, g, b)`` float tuple."""
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


#: Matplotlib-friendly version of :data:`PALETTE_HEX`.
PALETTE_RGB: dict[int, tuple[float, float, float]] = {
    k: hex_to_rgb(v) for k, v in PALETTE_HEX.items()
}


#: Pastel LaTeX-matching palette (``red!15``, ``teal!20``, ``gray!15``).
PALETTE_POSTER: dict[int, tuple[float, float, float]] = {
    0: (1.0, 0.86, 0.82),
    1: (0.80, 0.92, 0.90),
    2: (0.91, 0.91, 0.91),
}


#: Primary text (node labels, gene orders).
TEXT_COLOR: str = "#1f2937"
#: Tree edge strokes.
EDGE_COLOR: str = "#9ca3af"
#: Annotation / secondary text.
ANN_COLOR: str = "#6b7280"

LOSS_COLOR: str = "#dc2626"
GAIN_COLOR: str = "#16a34a"
INV_COLOR: str = "#7c3aed"

#: Font families.
FONT_SERIF: str = "serif"
FONT_MONO: str = "monospace"


__all__ = [
    "ANN_COLOR",
    "EDGE_COLOR",
    "FONT_MONO",
    "FONT_SERIF",
    "GAIN_COLOR",
    "INV_COLOR",
    "LOSS_COLOR",
    "PALETTE_HEX",
    "PALETTE_POSTER",
    "PALETTE_RGB",
    "TEXT_COLOR",
    "hex_to_rgb",
]