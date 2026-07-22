"""Brute-force counterexample search.
Enumerates Dollo-consistent loss scenarios on small phylogenetic trees,
builds the induced ternary matrix, and reports the ones that fail NB1P.
"""

from __future__ import annotations

import time
from typing import Dict, FrozenSet, List, Tuple

from nb1p._check import check_nb1p_fast

from sglpp.display import print_matrix, scenario_loss_labels
from sglpp.enumerate import enumerate_scenarios_pruned
from sglpp.scenario import Scenario
from sglpp.topology import generate_topologies
from sglpp.tree import number_edges


def _print_counterexample(
    sc: Scenario,
    matrix: List[List[int]],
    losses: Dict[int, FrozenSet[int]],
    idx: int,
):
    labels = scenario_loss_labels(sc)
    genomes = sc.leaf_genomes()
    print(f"\n  *** COUNTEREXAMPLE #{idx} ***  genes={sc.n_genes}")
    loss_str = {}
    for k, v in losses.items():
        if v:
            loss_str[k] = "{" + ",".join(sc.gene_name(g) for g in sorted(v)) + "}"
    print(f"  Losses: {loss_str}")
    print_matrix(matrix, sc.n_genes, labels)
    gen_str = {}
    for k, v in genomes.items():
        gen_str[k] = "{" + ",".join(sc.gene_name(g) for g in sorted(v)) + "}"
    print(f"  Leaf genomes: {gen_str}")


def search(
    max_leaves: int = 4,
    max_genes: int = 6,
    min_constraining: int = 2,
    max_total_losses: int = 99,
    visualize: bool = True,
    stop_after: int = 0,
):
    """Brute-force search for non-NB1P scenarios (optimized)."""
    counterexamples: List[Tuple[Scenario, List[List[int]]]] = []

    render = None
    if visualize:
        from sglpp.viz import visualize as render

    for n_leaves in range(3, max_leaves + 1):
        topos = generate_topologies(n_leaves)
        print(f"\n{'='*60}")
        print(f"  {n_leaves} leaves  →  {len(topos)} topolog{'y' if len(topos)==1 else 'ies'}")
        print(f"{'='*60}")

        for ti, tree in enumerate(topos):
            edges = number_edges(tree)
            print(f"\n  Topology {ti}: {tree}  "
                  f"({len(edges)} edges, depth {tree.depth()})")

            for n_genes in range(3, max_genes + 1):
                t0 = time.time()
                total = 0
                checked = 0
                found = 0

                for losses in enumerate_scenarios_pruned(
                    edges, n_genes,
                    min_constraining=min_constraining,
                    max_total_losses=max_total_losses,
                ):
                    total += 1
                    sc = Scenario(tree, n_genes, losses, edges)
                    matrix = sc.build_matrix()

                    ok, _ = check_nb1p_fast(matrix, n_genes)
                    checked += 1

                    if not ok:
                        found += 1
                        _print_counterexample(sc, matrix, losses,
                                              len(counterexamples) + 1)
                        counterexamples.append((sc, matrix))
                        if render is not None:
                            render(sc, matrix, len(counterexamples))

                        if stop_after > 0 and found >= stop_after:
                            break

                elapsed = time.time() - t0
                print(f"    genes={n_genes}: {total:,} scenarios "
                      f"({checked:,} checked) in {elapsed:.1f}s "
                      f"→ {found} counterexample(s)")

    print(f"\n{'='*60}")
    print(f"  TOTAL: {len(counterexamples)} counterexample(s) found")
    print(f"{'='*60}")
    return counterexamples
