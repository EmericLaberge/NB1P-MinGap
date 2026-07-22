"""4-state InOutParsimony DP engine — the parsimony core.

Extracted from :mod:`sglpp.sglpp.algorithm`: the dynamic-programming
machinery (the four cost tables c_min / c_in_extra / c_out_extra /
c_inout_extra, their :class:`_CostEntry` rows, and the recurrence) lives here,
apart from the tree-traversal pipeline.  Tree-free at runtime — it operates on
post-order node-id lists and gene-set dicts, with no
:class:`~sglpp.tree.PhyloTree` import.

The four DP states match the reference naming (Gascon, Delabre, El-Mabrouk
2026; port of https://github.com/UdeM-LBIT/InOutParsimony, commit e843c84):

* c_min       — node has no extra genes (only x_min)
* c_in_extra  — node has "in" extra genes (LCA below, premature presence)
* c_out_extra — node has "out" extra genes (absent from all descendant leaves)
* c_inout_extra — node has both
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from sglpp.tree import PhyloTree

#: Gene-set alias — ``frozenset`` of gene indices (int).
GeneSet = FrozenSet[int]

# ---------------------------------------------------------------------------
# DP cost entry — mirrors the reference's tuple structure.
# (cost, left_dict, (loss_left, gain_left), right_dict, (loss_right, gain_right))
# The dict points to the chosen child-table (c_min / c_in_extra / …).
# We use an enum-like string tag instead.
# ---------------------------------------------------------------------------

# Table tags (correspond to the four DP tables in the reference).
T_MIN = "min"
T_IN = "in_extra"
T_OUT = "out_extra"
T_INOUT = "inout_extra"


@dataclass
class _CostEntry:
    """One row of the DP cost table for a node."""

    cost: float = inf
    left_table: str = T_MIN
    left_loss: int = 0
    left_gain: int = 0
    right_table: str = T_MIN
    right_loss: int = 0
    right_gain: int = 0


def _run_dp(
    nodes_post: List[Tuple[int, "PhyloTree"]],
    x_min: Dict[int, GeneSet],
    delta_gain: int,
    delta_loss: int,
) -> Tuple[
    Dict[int, _CostEntry],
    Dict[int, _CostEntry],
    Dict[int, _CostEntry],
    Dict[int, _CostEntry],
]:
    """4-state DP (exact mirror of reference InOutParsimony).

    Returns (c_min, c_in, c_out, c_inout).
    """
    c_min: Dict[int, _CostEntry] = {}
    c_in: Dict[int, _CostEntry] = {}
    c_out: Dict[int, _CostEntry] = {}
    c_inout: Dict[int, _CostEntry] = {}

    def _loss(parent_nid: int, child_nid: int) -> int:
        return 0 if x_min[parent_nid] <= x_min[child_nid] else 1

    def _gain(parent_nid: int, child_nid: int) -> int:
        return 0 if x_min[child_nid] <= x_min[parent_nid] else 1

    # Helper to pick the best of 4 options.
    def _best4(options: list) -> _CostEntry:
        return min(options, key=lambda e: e.cost)

    def _delta_min(pnid: int, cnid: int) -> _CostEntry:
        g = _gain(pnid, cnid)
        l = _loss(pnid, cnid)
        opts = [
            _CostEntry(c_min[cnid].cost + delta_loss * l + delta_gain * g,
                        T_MIN, l, g, T_MIN, 0, 0),
            _CostEntry(c_in[cnid].cost + delta_loss * l + delta_gain,
                        T_IN, l, 1, T_IN, 0, 0),
            _CostEntry(c_out[cnid].cost + delta_gain * g + (0 if l else inf),
                        T_OUT, 0, g, T_OUT, 0, 0),
            _CostEntry(c_inout[cnid].cost + delta_gain + (0 if l else inf),
                        T_INOUT, 0, 1, T_INOUT, 0, 0),
        ]
        return _best4(opts)

    def _delta_out(pnid: int, cnid: int) -> _CostEntry:
        g = _gain(pnid, cnid)
        opts = [
            _CostEntry(c_min[cnid].cost + delta_loss + delta_gain * g,
                        T_MIN, 1, g, T_MIN, 0, 0),
            _CostEntry(c_in[cnid].cost + delta_loss + delta_gain,
                        T_IN, 1, 1, T_IN, 0, 0),
            _CostEntry(c_out[cnid].cost + delta_gain * g,
                        T_OUT, 0, g, T_OUT, 0, 0),
            _CostEntry(c_inout[cnid].cost + delta_gain,
                        T_INOUT, 0, 1, T_INOUT, 0, 0),
        ]
        return _best4(opts)

    def _delta_in(pnid: int, cnid: int) -> _CostEntry:
        g = _gain(pnid, cnid)
        l = _loss(pnid, cnid)
        opts = [
            _CostEntry(c_min[cnid].cost + delta_loss * l + (0 if g else inf),
                        T_MIN, l, 0, T_MIN, 0, 0),
            _CostEntry(c_in[cnid].cost + delta_loss * l,
                        T_IN, l, 0, T_IN, 0, 0),
            _CostEntry(c_out[cnid].cost + (0 if (g and l) else inf),
                        T_OUT, 0, 0, T_OUT, 0, 0),
            _CostEntry(c_inout[cnid].cost + (0 if l else inf),
                        T_INOUT, 0, 0, T_INOUT, 0, 0),
        ]
        return _best4(opts)

    def _delta_inout(pnid: int, cnid: int) -> _CostEntry:
        g = _gain(pnid, cnid)
        opts = [
            _CostEntry(c_min[cnid].cost + delta_loss + (0 if g else inf),
                        T_MIN, 1, 0, T_MIN, 0, 0),
            _CostEntry(c_in[cnid].cost + delta_loss,
                        T_IN, 1, 0, T_IN, 0, 0),
            _CostEntry(c_out[cnid].cost + (0 if g else inf),
                        T_OUT, 0, 0, T_OUT, 0, 0),
            _CostEntry(c_inout[cnid].cost,
                        T_INOUT, 0, 0, T_INOUT, 0, 0),
        ]
        return _best4(opts)

    def _dispatch_delta(pnid: int, cnid: int, table: str) -> _CostEntry:
        if table == T_MIN:
            return _delta_min(pnid, cnid)
        if table == T_OUT:
            return _delta_out(pnid, cnid)
        if table == T_IN:
            return _delta_in(pnid, cnid)
        return _delta_inout(pnid, cnid)

    for nid, node in nodes_post:
        if node.is_leaf():
            c_min[nid] = _CostEntry(cost=0)
            c_in[nid] = _CostEntry()
            c_out[nid] = _CostEntry()
            c_inout[nid] = _CostEntry()
        else:
            lnid = id(node.left)
            rnid = id(node.right)

            # ── c_min ──────────────────────────────────────────────
            if x_min[nid] != frozenset():
                dl = _delta_min(nid, lnid)
                dr = _delta_min(nid, rnid)
                c_min[nid] = _CostEntry(
                    dl.cost + dr.cost,
                    dl.left_table, dl.left_loss, dl.left_gain,
                    dr.left_table, dr.left_loss, dr.left_gain,
                )
            else:
                c_min[nid] = _CostEntry()

            # ── c_in_extra ─────────────────────────────────────────
            # Gains from left only
            lo_l = _delta_in(nid, lnid)
            lo_r = _delta_out(nid, rnid)
            c_in_lo = _CostEntry(
                lo_l.cost + lo_r.cost,
                lo_l.left_table, lo_l.left_loss, lo_l.left_gain,
                lo_r.left_table, lo_r.left_loss, lo_r.left_gain,
            )
            # Gains from right only
            ro_l = _delta_out(nid, lnid)
            ro_r = _delta_in(nid, rnid)
            c_in_ro = _CostEntry(
                ro_l.cost + ro_r.cost,
                ro_l.left_table, ro_l.left_loss, ro_l.left_gain,
                ro_r.left_table, ro_r.left_loss, ro_r.left_gain,
            )
            # Gains from both sides
            bs_l = _delta_inout(nid, lnid)
            bs_r = _delta_inout(nid, rnid)
            c_in_bs = _CostEntry(
                bs_l.cost + bs_r.cost,
                bs_l.left_table, bs_l.left_loss, bs_l.left_gain,
                bs_r.left_table, bs_r.left_loss, bs_r.left_gain,
            )
            c_in[nid] = min(c_in_lo, c_in_ro, c_in_bs, key=lambda e: e.cost)

            # ── c_out_extra ────────────────────────────────────────
            ol = _delta_out(nid, lnid)
            orr = _delta_out(nid, rnid)
            c_out[nid] = _CostEntry(
                ol.cost + orr.cost,
                ol.left_table, ol.left_loss, ol.left_gain,
                orr.left_table, orr.left_loss, orr.left_gain,
            )

            # ── c_inout_extra ──────────────────────────────────────
            # Left only
            io_lo_l = _delta_inout(nid, lnid)
            io_lo_r = _delta_out(nid, rnid)
            c_io_lo = _CostEntry(
                io_lo_l.cost + io_lo_r.cost,
                io_lo_l.left_table, io_lo_l.left_loss, io_lo_l.left_gain,
                io_lo_r.left_table, io_lo_r.left_loss, io_lo_r.left_gain,
            )
            # Right only
            io_ro_l = _delta_out(nid, lnid)
            io_ro_r = _delta_inout(nid, rnid)
            c_io_ro = _CostEntry(
                io_ro_l.cost + io_ro_r.cost,
                io_ro_l.left_table, io_ro_l.left_loss, io_ro_l.left_gain,
                io_ro_r.left_table, io_ro_r.left_loss, io_ro_r.left_gain,
            )
            # Both
            io_bs_l = _delta_inout(nid, lnid)
            io_bs_r = _delta_inout(nid, rnid)
            c_io_bs = _CostEntry(
                io_bs_l.cost + io_bs_r.cost,
                io_bs_l.left_table, io_bs_l.left_loss, io_bs_l.left_gain,
                io_bs_r.left_table, io_bs_r.left_loss, io_bs_r.left_gain,
            )
            c_inout[nid] = min(c_io_lo, c_io_ro, c_io_bs, key=lambda e: e.cost)

    return c_min, c_in, c_out, c_inout
