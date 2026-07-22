"""Phylogenetics: Dollo loss scenarios on rooted binary trees.
Builds ternary matrices from gene-loss scenarios and searches small trees
for NB1P counterexamples.
Submodules:
    tree        — PhyloTree, EdgeInfo, number_edges, parse_tree
    topology    — generate_topologies
    scenario    — Scenario (Dollo-consistent loss model)
    enumerate   — enumerate_scenarios / enumerate_scenarios_pruned
    display     — print_matrix, scenario_loss_labels, GENE_NAMES
    search      — NB1P-counterexample brute-force search (pruned, fast)
    viz         — visualize (matplotlib PNG; needs the viz extra)
    annotated_tree — RichText/EdgeLabel/EdgeMarker + TreeRenderer
    layout      — TreeLayout (shared ortho-cladogram coordinates)
    edge_events — draw_edge_events (× loss / • gain / ↺ inversion markers)
    render      — palette + labels (single source of truth for colors / text)
"""

from sglpp._search import search
from sglpp.annotated_tree import AnnotatedTree, EdgeLabel, EdgeMarker, RichText
from sglpp.tree_renderer import TreeRenderer
from sglpp.enumerate import (
    enumerate_scenarios,
    enumerate_scenarios_pruned,
)
from sglpp.scenario import GENE_NAMES, Scenario
from sglpp.topology import generate_topologies
from sglpp.tree import EdgeInfo, PhyloTree, number_edges, parse_tree

__all__ = [
    "AnnotatedTree",
    "EdgeInfo",
    "EdgeLabel",
    "EdgeMarker",
    "GENE_NAMES",
    "PhyloTree",
    "RichText",
    "Scenario",
    "TreeRenderer",
    "enumerate_scenarios",
    "enumerate_scenarios_pruned",
    "generate_topologies",
    "number_edges",
    "parse_tree",
    "search",
]
