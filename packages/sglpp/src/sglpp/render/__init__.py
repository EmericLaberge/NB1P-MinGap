"""Render subpackage — single source of truth for all rendering primitives.

Layout:

- :mod:`sglpp.render.palette` — every color / font constant.
- :mod:`sglpp.render.labels`  — gene-set / order text helpers.
- :mod:`sglpp.render.matrix`  — :func:`draw_matrix` (heat map).
- :mod:`sglpp.render.tree`    — :func:`render_mingap`.
"""

from sglpp.render.labels import (
    gene_label,
    order_str,
    setstr,
)
from sglpp.render.matrix import draw_matrix
from sglpp.render.palette import (
    ANN_COLOR,
    EDGE_COLOR,
    FONT_MONO,
    FONT_SERIF,
    GAIN_COLOR,
    INV_COLOR,
    LOSS_COLOR,
    PALETTE_HEX,
    PALETTE_POSTER,
    PALETTE_RGB,
    TEXT_COLOR,
    hex_to_rgb,
)
from sglpp.render.tree import render_mingap

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
    "draw_matrix",
    "gene_label",
    "hex_to_rgb",
    "order_str",
    "render_mingap",
    "setstr",
]