#!/usr/bin/env python3
"""Render a combined SGLPP tree + MinGap matrix for a scenarios_sglpp row.

Usage:
    uv run python scripts/render_sglpp_mingap.py <scenario_id>
"""

from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import duckdb

from mingap.score import block_cost
from sglpp import parse_tree
from sglpp.render.tree import render_mingap
from sglpp.sglpp.algorithm import solve


def load_row(db_path: str, scenario_id: int):
    con = duckdb.connect(db_path)
    row = con.execute(
        """
        SELECT id, orig_id, n_leaves, n_genes, tree, leaf_sets, gains, losses, matrix, cost
        FROM scenarios_sglpp
        WHERE id = ?
        """,
        [scenario_id],
    ).fetchone()
    con.close()
    if row is None:
        raise SystemExit(f"scenario {scenario_id} not found")
    return row


def build_leaf_orders(tree, leaf_sets):
    orders = {}
    stack = [tree]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            orders[id(node)] = sorted(leaf_sets.get(node.label, []))
        else:
            stack.extend([node.left, node.right])
    return orders


def main() -> int:
    scenario_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1705
    out = Path(".tmp") / f"sglpp_mingap_{scenario_id}.png"
    out.parent.mkdir(exist_ok=True)

    row = load_row("data/scenarios_sglpp.duckdb", scenario_id)
    (
        id_,
        orig_id,
        n_leaves,
        n_genes,
        tree_str,
        leaf_sets_json,
        _,
        matrix_json,
        cost,
    ) = row
    leaf_sets = {int(k): set(v) for k, v in json.loads(leaf_sets_json).items()}
    matrix = json.loads(matrix_json)

    tree = parse_tree(tree_str)
    res = solve(tree, leaf_sets, n_genes)

    # brute force all permutations
    costs = sorted(
        (block_cost(matrix, list(p)), p) for p in permutations(range(n_genes))
    )
    best_cost, best_perm = costs[0]

    print(f"scenario id={id_} orig_id={orig_id}")
    print(f"tree = {tree_str}")
    print(f"leaves = {n_leaves}, genes = {n_genes}")
    print(f"stored SGLPP cost = {cost}, recomputed = {res.cost}")
    print(f"gains = {res.to_gain_dict()}")
    print(f"losses = {res.to_loss_dict()}")
    print(f"leaf_sets = {leaf_sets}")
    print(f"\nbrute force {len(costs)} permutations:")
    for c, p in costs:
        print(f"  cost={c}  perm={p}")
    print(f"\nBest: cost={best_cost}  perm={best_perm}")

    leaf_orders = build_leaf_orders(tree, leaf_sets)
    render_mingap(tree, res, leaf_orders, list(best_perm), best_cost, str(out))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
