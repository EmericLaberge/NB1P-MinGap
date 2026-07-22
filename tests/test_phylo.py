"""Tests for the phylogenetics submodule (trees, scenarios, search)."""

from __future__ import annotations

import nb1p
import sglpp

from sglpp import (
    PhyloTree,
    Scenario,
    enumerate_scenarios,
    generate_topologies,
    number_edges,
)


def test_topology_counts():
    # Wedderburn–Etherington: unlabelled rooted binary shapes.
    assert len(generate_topologies(3)) == 1
    assert len(generate_topologies(4)) == 2
    assert len(generate_topologies(5)) == 3


def test_tree_leaves_and_edges():
    tree = PhyloTree.node(
        PhyloTree.leaf(0),
        PhyloTree.node(PhyloTree.leaf(1), PhyloTree.leaf(2)),
    )
    assert sorted(tree.leaves()) == [0, 1, 2]
    edges = number_edges(tree)
    # A rooted binary tree with 3 leaves has 4 edges.
    assert len(edges) == 4


def test_scenario_build_matrix_is_ternary():
    tree = generate_topologies(3)[0]
    edges = number_edges(tree)
    losses = next(enumerate_scenarios(edges, 3))
    sc = Scenario(tree, 3, losses, edges)
    matrix = sc.build_matrix()
    for row in matrix:
        assert all(v in (0, 1, 2) for v in row)


def test_search_finds_counterexamples():
    # Small exhaustive search must turn up at least one NB1P failure.
    found = sglpp.search(
        max_leaves=4, max_genes=5, min_constraining=2,
        visualize=False, stop_after=1,
    )
    assert len(found) >= 1
    for _sc, matrix in found:
        # A counter-example: no permutation satisfies NB1P.
        assert not nb1p.check(matrix)
