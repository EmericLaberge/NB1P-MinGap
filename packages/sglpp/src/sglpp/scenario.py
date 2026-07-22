"""Dollo-consistent loss scenario on a phylogenetic tree."""

from __future__ import annotations

from typing import Dict, FrozenSet, List

from sglpp.tree import EdgeInfo, PhyloTree

GENE_NAMES = "abcdefghijklmnopqrstuvwxyz"


class Scenario:
    """Dollo-consistent loss scenario on a phylogenetic tree.
    Each gene is gained once at the root and can be lost independently
    on different branches. On any root-to-leaf path a gene is lost at
    most once (enforced by construction).
    """

    def __init__(
        self,
        tree: PhyloTree,
        n_genes: int,
        losses: Dict[int, FrozenSet[int]],
        edges: List[EdgeInfo],
    ):
        self.tree = tree
        self.n_genes = n_genes
        self.losses = losses  # edge_id → lost gene-set
        self.edges = edges

    def present_at(self, edge_id: int) -> FrozenSet[int]:
        """Gene-set present at the *parent* of this edge."""
        all_g = frozenset(range(self.n_genes))
        lost: FrozenSet[int] = frozenset()
        for a in self.edges[edge_id].ancestors:
            lost = lost | self.losses.get(a, frozenset())
        return all_g - lost

    def build_matrix(self) -> List[List[int]]:
        """Ternary matrix (rows = edges with non-empty losses).

        Values: ``0`` = present, ``1`` = lost here, ``2`` = already lost by
        an ancestor.
        """
        rows: List[List[int]] = []
        for ei in self.edges:
            loss = self.losses.get(ei.id, frozenset())
            if not loss:
                continue
            present = self.present_at(ei.id)
            rows.append(
                [
                    1 if g in loss else (0 if g in present else 2)
                    for g in range(self.n_genes)
                ]
            )
        return rows

    def leaf_genomes(self) -> Dict[int, FrozenSet[int]]:
        """Present gene-set at each leaf."""
        out: Dict[int, FrozenSet[int]] = {}
        for ei in self.edges:
            if ei.child.is_leaf():
                out[ei.child.label] = self.present_at(ei.id) - self.losses.get(
                    ei.id, frozenset()
                )
        return out

    def gene_name(self, g: int) -> str:
        return GENE_NAMES[g] if g < len(GENE_NAMES) else str(g)
