"""SGLPP result type — the public output contract of the parsimony solve.

Separated from :mod:`sglpp.sglpp.algorithm` so the result accessors
(notably :meth:`SGLPPResult.build_matrix`, which converts to the downstream
NB1P ternary matrix) live apart from the InOutParsimony DP internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

from sglpp.tree import EdgeInfo, PhyloTree

__all__ = ["SGLPPResult"]


@dataclass
class SGLPPResult:
    """Complete result of an InOutParsimony solve."""

    tree: PhyloTree
    edges: List[EdgeInfo]
    n_genes: int
    cost: int
    gains: Dict[int, FrozenSet[int]]
    losses: Dict[int, FrozenSet[int]]
    node_contents: Dict[int, FrozenSet[int]]
    x_min: Dict[int, FrozenSet[int]]
    x_lca: Dict[int, FrozenSet[int]]
    pgl: Dict[int, Tuple[bool, bool]]
    delta_gain: int
    delta_loss: int

    # ── Convenience accessors ──────────────────────────────────────

    def leaf_genomes(self) -> Dict[int, FrozenSet[int]]:
        """Gene content at each leaf (from reconstructed node_contents)."""
        out: Dict[int, FrozenSet[int]] = {}
        for e in self.edges:
            if e.child.is_leaf():
                out[e.child.label] = self.node_contents.get(id(e.child), frozenset())
        return out

    def to_gain_dict(self) -> Dict[int, List[int]]:
        return {eid: sorted(gs) for eid, gs in self.gains.items() if gs}

    def to_loss_dict(self) -> Dict[int, List[int]]:
        return {eid: sorted(gs) for eid, gs in self.losses.items() if gs}

    def build_matrix(self) -> list[list[int]]:
        """Build the combined NB1P matrix ``{0,1,2}^{2E×G}``.

        First *E* rows = loss view, next *E* rows = gain view.
        Same encoding for both — ``block_cost`` counts blocks of ``2``:

        **Loss rows** (per edge, per gene):
          - ``0`` = present at child node
          - ``1`` = lost on this edge
          - ``2`` = absent at child (already lost or never gained)

        **Gain rows** (per edge, per gene):
          - ``0`` = present at parent node (already gained)
          - ``1`` = gained on this edge
          - ``2`` = absent at parent (not yet gained)
        """
        rows: list[list[int]] = []
        for e in self.edges:
            child_content = self.node_contents.get(id(e.child), frozenset())
            parent_content = self.node_contents.get(id(e.parent), frozenset())
            eg = self.gains.get(e.id, frozenset())
            el = self.losses.get(e.id, frozenset())
            loss_row: list[int] = []
            gain_row: list[int] = []
            for g in range(self.n_genes):
                loss_row.append(1 if g in el else (0 if g in child_content else 2))
                gain_row.append(1 if g in eg else (0 if g in parent_content else 2))
            rows.append(loss_row)
            rows.append(gain_row)
        return rows

    def n_gain_edges(self) -> int:
        return sum(1 for gs in self.gains.values() if gs)

    def n_loss_edges(self) -> int:
        return sum(1 for gs in self.losses.values() if gs)

    def summary(self) -> str:
        return (
            f"SGLPPResult(cost={self.cost}, "
            f"gains={self.n_gain_edges()}, losses={self.n_loss_edges()})"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()
