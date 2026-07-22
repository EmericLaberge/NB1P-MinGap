"""Terminal pretty-print helpers and shared constants."""

from __future__ import annotations

from typing import List, Optional

from sglpp.scenario import GENE_NAMES, Scenario


def print_matrix(
    matrix: List[List[int]], n_genes: int, loss_labels: Optional[List[str]] = None
):
    hdr = "     " + "  ".join(GENE_NAMES[g] for g in range(n_genes))
    print(hdr)
    for i, row in enumerate(matrix):
        lbl = loss_labels[i] if loss_labels and i < len(loss_labels) else f"r{i}"
        vals = "  ".join(str(v) for v in row)
        print(f"  {lbl}: {vals}")


def scenario_loss_labels(scenario: Scenario) -> List[str]:
    """One label per non-empty-loss edge."""
    labels: List[str] = []
    for ei in scenario.edges:
        loss = scenario.losses.get(ei.id, frozenset())
        if not loss:
            continue
        genes = ",".join(scenario.gene_name(g) for g in sorted(loss))
        labels.append(f"L{ei.id}:{{{genes}}}")
    return labels
