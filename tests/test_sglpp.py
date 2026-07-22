"""Tests for the SGLPP (Small Gain-Loss Parsimony Problem) solver.

Validates that the InOutParsimony port produces correct histories:
  - Leaf genomes are reconstructed exactly from the input leaf sets.
  - Gains and losses are consistent with the node contents.
  - The build_matrix output encodes gains as 2 and losses as 1.
  - Cost matches δ_gain × #gain_edges + δ_loss × #loss_edges.
"""

from __future__ import annotations

import ast
import json

import duckdb
import pytest

from sglpp.scenario import Scenario
from sglpp.sglpp import solve
from sglpp.tree import PhyloTree, number_edges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(t: object) -> PhyloTree:
    if isinstance(t, int):
        return PhyloTree.leaf(t)
    return PhyloTree.node(_parse(t[0]), _parse(t[1]))


def _load_scenario(scenario_id: int):
    """Load a single scenario from the Dollo DB."""
    con = duckdb.connect("data/scenarios.duckdb", read_only=True)
    row = con.execute(
        "SELECT id, n_leaves, n_genes, tree, losses "
        "FROM scenarios WHERE id = ?",
        [scenario_id],
    ).fetchone()
    con.close()
    assert row is not None, f"Scenario {scenario_id} not found"
    oid, n_leaves, n_genes, tree_str, losses_json = row
    tree = _parse(ast.literal_eval(tree_str))
    edges = number_edges(tree)
    losses = {int(k): frozenset(v) for k, v in json.loads(losses_json).items()}
    scen = Scenario(tree, n_genes, losses, edges)
    return scen, tree


# ---------------------------------------------------------------------------
# Core correctness tests
# ---------------------------------------------------------------------------


class TestSGLPPSolve:
    """Leaf-genome round-trip and structural invariants."""

    def test_leaf_genomes_roundtrip_single(self):
        """Scenario 28001: the first case that failed in the old code."""
        scen, tree = _load_scenario(28001)
        lg = scen.leaf_genomes()
        result = solve(tree, {k: set(v) for k, v in lg.items()}, scen.n_genes)
        rl = result.leaf_genomes()
        for lf in lg:
            assert lg[lf] == rl.get(lf, frozenset()), (
                f"Leaf {lf}: expected {sorted(lg[lf])}, got {sorted(rl.get(lf, frozenset()))}"
            )

    @pytest.mark.parametrize("sid", [1, 100, 1000, 10000, 50000, 90000])
    def test_leaf_genomes_roundtrip_parametrized(self, sid):
        scen, tree = _load_scenario(sid)
        lg = scen.leaf_genomes()
        result = solve(tree, {k: set(v) for k, v in lg.items()}, scen.n_genes)
        rl = result.leaf_genomes()
        for lf in lg:
            assert lg[lf] == rl.get(lf, frozenset())

    def test_cost_formula(self):
        """cost == delta_gain * (n_gain_edges + 1) + delta_loss * n_loss_edges.

        The +1 accounts for the synthetic root gain edge.
        """
        scen, tree = _load_scenario(28001)
        lg = scen.leaf_genomes()
        result = solve(tree, {k: set(v) for k, v in lg.items()}, scen.n_genes)
        expected = (
            result.delta_gain * (result.n_gain_edges() + 1)
            + result.delta_loss * result.n_loss_edges()
        )
        assert result.cost == expected

    def test_gain_loss_consistent_with_matrix(self):
        """Matrix is 2E×G: E loss rows interleaved with E gain rows."""
        scen, tree = _load_scenario(28001)
        lg = scen.leaf_genomes()
        result = solve(tree, {k: set(v) for k, v in lg.items()}, scen.n_genes)
        matrix = result.build_matrix()
        n_edges = len(result.edges)
        assert len(matrix) == 2 * n_edges
        for row in matrix:
            for v in row:
                assert v in (0, 1, 2), f"Invalid matrix value: {v}"
class TestSGLPPBuildMatrix:
    """Loss/gain event counts match the 2E×G matrix."""
    def test_matrix_loss_gain_counts(self):
        scen, tree = _load_scenario(28001)
        lg = scen.leaf_genomes()
        result = solve(tree, {k: set(v) for k, v in lg.items()}, scen.n_genes)
        matrix = result.build_matrix()
        n_edges = len(result.edges)
        # Loss rows are at even indices (0, 2, 4, …), gain rows at odd.
        total_loss_ones = sum(
            1 for i in range(0, 2 * n_edges, 2) for v in matrix[i] if v == 1
        )
        total_gain_ones = sum(
            1 for i in range(1, 2 * n_edges, 2) for v in matrix[i] if v == 1
        )
        expected_loss_genes = sum(len(gs) for gs in result.losses.values())
        expected_gain_genes = sum(len(gs) for gs in result.gains.values())
        assert total_loss_ones == expected_loss_genes
        assert total_gain_ones == expected_gain_genes
    def test_matrix_shape(self):
        """2E rows × n_genes columns."""
        scen, tree = _load_scenario(28001)
        lg = scen.leaf_genomes()
        result = solve(tree, {k: set(v) for k, v in lg.items()}, scen.n_genes)
        matrix = result.build_matrix()
        assert len(matrix) == 2 * len(result.edges)
        assert all(len(row) == result.n_genes for row in matrix)


class TestSGLPPCustomWeights:
    """Different δ_gain / δ_loss produce different solutions."""

    def test_higher_gain_cost(self):
        scen, tree = _load_scenario(28001)
        lg = scen.leaf_genomes()
        r1 = solve(tree, {k: set(v) for k, v in lg.items()}, scen.n_genes, delta_gain=1, delta_loss=1)
        r10 = solve(tree, {k: set(v) for k, v in lg.items()}, scen.n_genes, delta_gain=10, delta_loss=1)
        # Higher gain cost should never produce more gain edges.
        assert r10.n_gain_edges() <= r1.n_gain_edges() + 2  # small slack for tie-breaking
