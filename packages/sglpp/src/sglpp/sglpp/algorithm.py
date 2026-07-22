"""SGLPP algorithm — port of InOutParsimony (Gascon, Delabre, El-Mabrouk 2026).
Line-for-line logical equivalent of the reference at
https://github.com/UdeM-LBIT/InOutParsimony (commit e843c84), adapted
from sowing.Zipper navigation to our PhyloTree objects keyed by id(node).
The four DP states match the reference naming:
* c_min       — node has no extra genes (only x_min)
* c_in_extra  — node has "in" extra genes (LCA below, premature presence)
* c_out_extra — node has "out" extra genes (absent from all descendant leaves)
* c_inout_extra — node has both
Reconstruction follows the reference's two-pass x_content() algorithm:
  Pass 1 (post-order): inherit child content when there is no gain event.
  Pass 2 (pre-order):  inherit parent content when there is no loss event.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Set, Tuple

from sglpp.tree import EdgeInfo, PhyloTree, number_edges
from sglpp.sglpp._dp import (
    GeneSet,
    T_IN,
    T_INOUT,
    T_MIN,
    T_OUT,
    _CostEntry,
    _run_dp,
)
from sglpp.sglpp.result import SGLPPResult

# Default unit costs.
DEFAULT_DELTA_GAIN: int = 1
DEFAULT_DELTA_LOSS: int = 1


# ---------------------------------------------------------------------------
# Step helpers — each step of the algorithm extracted for clarity.
# ---------------------------------------------------------------------------


def _collect_postorder(tree: PhyloTree) -> List[Tuple[int, PhyloTree]]:
    """Step 0: collect nodes in post-order (left, right, root)."""
    nodes_post: List[Tuple[int, PhyloTree]] = []

    def _collect(t: PhyloTree) -> None:
        if t.is_leaf():
            nodes_post.append((id(t), t))
        else:
            _collect(t.left)
            _collect(t.right)
            nodes_post.append((id(t), t))

    _collect(tree)
    return nodes_post


def _compute_lca_content(
    tree: PhyloTree,
    leaf_sets: Dict[int, Set[int]],
    nodes_post: List[Tuple[int, PhyloTree]],
) -> Tuple[Dict[int, GeneSet], Dict[int, GeneSet]]:
    """Step 1: LCA_Content.

    Returns (A, x_lca) where:
      * A[v]      = union of all leaf-sets in the subtree of v.
      * x_lca[v]  = genes whose LCA is exactly v.
    """
    # A[v] = union of all leaf-sets in the subtree of v.
    A: Dict[int, GeneSet] = {}
    for nid, node in nodes_post:
        if node.is_leaf():
            A[nid] = frozenset(leaf_sets.get(node.label, set()))
        else:
            A[nid] = A[id(node.left)] | A[id(node.right)]

    # B[root] = A[root];  B[child] = B[parent] - A[sibling].
    # x_lca[v] = genes whose LCA is exactly v.
    B: Dict[int, GeneSet] = {}
    x_lca: Dict[int, GeneSet] = {}
    B[id(tree)] = A[id(tree)]

    # Pre-order traversal for B and x_lca.
    def _preorder(node: PhyloTree) -> None:
        if node.is_leaf():
            x_lca[id(node)] = B[id(node)]
            return
        left, right = node.left, node.right
        B[id(left)] = B[id(node)] - A[id(right)]
        B[id(right)] = B[id(node)] - A[id(left)]
        x_lca[id(node)] = (B[id(node)] - B[id(left)]) - B[id(right)]
        _preorder(left)
        _preorder(right)

    _preorder(tree)
    # Leaves that weren't visited by _preorder (they're visited above).
    for nid, node in nodes_post:
        if node.is_leaf():
            x_lca[id(node)] = B.get(id(node), frozenset())

    return A, x_lca


def _compute_min_content(
    nodes_post: List[Tuple[int, PhyloTree]],
    leaf_sets: Dict[int, Set[int]],
    x_lca: Dict[int, GeneSet],
) -> Dict[int, GeneSet]:
    """Step 2: Min_Content — compute x_min."""
    x_min: Dict[int, GeneSet] = {}
    for nid, node in nodes_post:
        if node.is_leaf():
            x_min[nid] = frozenset(leaf_sets.get(node.label, set()))
        else:
            x_min[nid] = (
                (x_min[id(node.left)] - x_lca[id(node.left)])
                | (x_min[id(node.right)] - x_lca[id(node.right)])
            )
    return x_min


def _solve_root(
    tree: PhyloTree,
    c_min: Dict[int, _CostEntry],
    c_in: Dict[int, _CostEntry],
    delta_gain: int,
) -> Tuple[int, _CostEntry, int]:
    """Step 4: root solution.

    Returns (root_nid, min_sol, cost).
    """
    root_nid = id(tree)
    min_sol = min(c_min[root_nid], c_in[root_nid], key=lambda e: e.cost)
    cost = min_sol.cost + delta_gain
    return root_nid, min_sol, cost


def _backtrack_pgl(
    tree: PhyloTree,
    root_nid: int,
    min_sol: _CostEntry,
    c_min: Dict[int, _CostEntry],
    c_in: Dict[int, _CostEntry],
    c_out: Dict[int, _CostEntry],
    c_inout: Dict[int, _CostEntry],
) -> Dict[int, Tuple[bool, bool]]:
    """Step 5: pGainLoss — backtrack (loss, gain) per node."""
    tables = {T_MIN: c_min, T_IN: c_in, T_OUT: c_out, T_INOUT: c_inout}

    # positionGainLoss: node_id → (has_loss: bool, has_gain: bool)
    pgl: Dict[int, Tuple[bool, bool]] = {}
    pgl[root_nid] = (False, True)  # root always has gain (synthetic edge)

    def _propagate(node: PhyloTree, entry: _CostEntry) -> None:
        if node.is_leaf():
            return
        lnid = id(node.left)
        rnid = id(node.right)

        left_entry = tables[entry.left_table][lnid]
        right_entry = tables[entry.right_table][rnid]

        pgl[lnid] = (bool(entry.left_loss), bool(entry.left_gain))
        pgl[rnid] = (bool(entry.right_loss), bool(entry.right_gain))

        _propagate(node.left, left_entry)
        _propagate(node.right, right_entry)

    _propagate(tree, min_sol)
    return pgl


def _reconstruct_content(
    tree: PhyloTree,
    nodes_post: List[Tuple[int, PhyloTree]],
    x_min: Dict[int, GeneSet],
    pgl: Dict[int, Tuple[bool, bool]],
) -> Dict[int, Set[int]]:
    """Step 6: x_content — two-pass reconstruction."""
    x: Dict[int, Set[int]] = {}


    for nid, node in nodes_post:
        if node.is_leaf():
            x[nid] = set(x_min[nid])
        else:
            left_loss, left_gain = pgl.get(id(node.left), (False, False))
            right_loss, right_gain = pgl.get(id(node.right), (False, False))
            x_l = x[id(node.left)] if not left_gain else set()
            x_r = x[id(node.right)] if not right_gain else set()
            x[nid] = set(x_min[nid]) | x_l | x_r


    def _preorder_x(node: PhyloTree) -> None:
        if node.is_leaf():
            return
        for child in (node.left, node.right):
            cnid = id(child)
            child_loss, _ = pgl.get(cnid, (False, False))
            if not child_loss:
                x[cnid] = x[cnid] | x[id(node)]
            _preorder_x(child)

    _preorder_x(tree)
    return x


def _build_edge_events(
    tree: PhyloTree,
    x: Dict[int, Set[int]],
) -> Tuple[List[EdgeInfo], Dict[int, FrozenSet[int]], Dict[int, FrozenSet[int]]]:
    """Step 7: build per-edge gains/losses + edge numbering.

    Returns (edges, gains, losses).
    """
    edges = number_edges(tree)

    gains: Dict[int, FrozenSet[int]] = {}
    losses: Dict[int, FrozenSet[int]] = {}

    for e in edges:
        parent_x = frozenset(x.get(id(e.parent), set()))
        child_x = frozenset(x.get(id(e.child), set()))
        edge_gains = child_x - parent_x
        edge_losses = parent_x - child_x
        gains[e.id] = edge_gains
        losses[e.id] = edge_losses

    return edges, gains, losses




def solve(
    tree: PhyloTree,
    leaf_sets: Dict[int, Set[int]],
    n_genes: int,
    *,
    delta_gain: int = DEFAULT_DELTA_GAIN,
    delta_loss: int = DEFAULT_DELTA_LOSS,
) -> "SGLPPResult":
    """Solve the Small Gain-Loss Parsimony Problem on *tree*.

    Args:
        tree: rooted binary phylogenetic tree.
        leaf_sets: ``leaf_label → set[gene]`` — gene content of each leaf.
        n_genes: total number of gene families.
        delta_gain: cost per gain edge (default 1).
        delta_loss: cost per loss edge (default 1).

    Returns:
        :class:`SGLPPResult` with cost, node contents, gains, losses per edge.
    """
    # ── Step 0: collect nodes in post-order ────────────────────────────
    nodes_post = _collect_postorder(tree)

    # ── Step 1: LCA_Content ────────────────────────────────────────────
    A, x_lca = _compute_lca_content(tree, leaf_sets, nodes_post)

    # ── Step 2: Min_Content ────────────────────────────────────────────
    x_min = _compute_min_content(nodes_post, leaf_sets, x_lca)

    # ── Step 3: 4-state DP (exact mirror of reference InOutParsimony) ─
    c_min, c_in, c_out, c_inout = _run_dp(
        nodes_post, x_min, delta_gain, delta_loss
    )

    # ── Step 4: root solution ──────────────────────────────────────────
    root_nid, min_sol, cost = _solve_root(tree, c_min, c_in, delta_gain)

    # ── Step 5: pGainLoss — backtrack (loss, gain) per node ────────────
    pgl = _backtrack_pgl(
        tree, root_nid, min_sol, c_min, c_in, c_out, c_inout
    )

    # ── Step 6: x_content — two-pass reconstruction ────────────────────
    x = _reconstruct_content(tree, nodes_post, x_min, pgl)

    # ── Step 7: build per-edge gains/losses + edge numbering ───────────
    edges, gains, losses = _build_edge_events(tree, x)


    node_contents: Dict[int, FrozenSet[int]] = {
        nid: frozenset(gs) for nid, gs in x.items()
    }

    return SGLPPResult(
        tree=tree,
        edges=edges,
        n_genes=n_genes,
        cost=cost,
        gains=gains,
        losses=losses,
        node_contents=node_contents,
        x_min=x_min,
        x_lca=x_lca,
        pgl=pgl,
        delta_gain=delta_gain,
        delta_loss=delta_loss,
    )

