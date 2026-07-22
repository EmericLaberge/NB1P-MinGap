"""Tests for the backend-agnostic core (verify, block_cost, encoding, validation)."""

from __future__ import annotations

import pytest

import nb1p
from nb1p import (
    extract_perm,
    num_pairs,
    pair_index,
    validate_ternary,
)
from mingap.score import block_cost


def test_validate_ternary_dims(paper_matrix):
    assert validate_ternary(paper_matrix) == (6, 5)


def test_validate_ternary_rejects_ragged():
    with pytest.raises(ValueError):
        validate_ternary([[0, 1], [0]])


def test_validate_ternary_rejects_non_ternary():
    with pytest.raises(ValueError):
        validate_ternary([[0, 1, 3]])


def test_pair_index_is_bijective():
    n = 6
    seen = set()
    for a in range(n):
        for b in range(a + 1, n):
            idx = pair_index(a, b, n)
            assert 0 <= idx < num_pairs(n)
            seen.add(idx)
    assert len(seen) == num_pairs(n)


def test_verify_accepts_valid_permutation():
    matrix = [[1, 1, 0], [0, 1, 1]]
    assert nb1p.verify(matrix, [0, 1, 2])


def test_verify_rejects_broken_permutation():
    matrix = [[1, 0, 1]]
    assert not nb1p.verify(matrix, [0, 1, 2])
    assert nb1p.verify(matrix, [0, 2, 1])


# ── block_cost ──────────────────────────────────────────────────


def test_block_cost_zero_when_no_blocks():
    # Single 1 → no block to count
    assert block_cost([[1, 0, 0]], [0, 1, 2]) == 0


def test_block_cost_single_contiguous_block():
    # Two 1s adjacent → 1 block, cost 0
    assert block_cost([[1, 1, 0]], [0, 1, 2]) == 0


def test_block_cost_two_blocks():
    # 1, 0, 1 → two blocks of {1}, cost 1
    assert block_cost([[1, 0, 1]], [0, 1, 2]) == 1


def test_block_cost_2_glues_1s():
    # 1, 2, 1 → one block (2 acts as glue), cost 0
    assert block_cost([[1, 2, 1]], [0, 1, 2]) == 0


def test_block_cost_isolated_2_ignored():
    # 2, 0, 2 → two 2-only blocks, neither counts, cost 0
    assert block_cost([[2, 0, 2]], [0, 1, 2]) == 0


def test_block_cost_multi_row():
    M = [
        [1, 0, 1, 0, 1],  # 3 blocks with 1s → cost 2
        [1, 2, 0, 2, 1],  # 2 blocks with 1s → cost 1
        [0, 0, 0, 0, 0],  # no losses → cost 0
    ]
    assert block_cost(M, [0, 1, 2, 3, 4]) == 3


def test_block_cost_top_level():
    M = nb1p.paper_example()
    perm = list(range(5))
    from mingap.score import block_cost as core_block_cost
    assert core_block_cost(M, perm) == block_cost(M, perm)


def test_extract_perm_roundtrip():
    n = 3
    var_val = {
        pair_index(0, 1, n): 1.0,
        pair_index(0, 2, n): 1.0,
        pair_index(1, 2, n): 1.0,
    }
    assert extract_perm(var_val, n) == [0, 1, 2]
