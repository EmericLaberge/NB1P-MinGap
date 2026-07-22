"""Tests for the real biological data loader (bench/real_data.py).

Covers:
  - count-to-ternary mapping
  - parser handles the TSV header / Total column
  - small-tier loader (``load_real_matrices``) degrades gracefully when
    the cache + network both fail
  - large-tier loader (``load_large_matrices``) degrades gracefully when
    the cache + network both fail
  - combined loader (``load_all_real``) returns both tiers concatenated
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
REAL_DATA_PATH = BENCH / "real_data.py"
spec = importlib.util.spec_from_file_location("real_data", REAL_DATA_PATH)
assert spec and spec.loader, "could not load bench/real_data.py"
real_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real_data)


def test_count_to_ternary_mapping():
    """0 -> lost (1), 1 -> present (0), >=2 -> wildcard (2)."""
    assert real_data._count_to_ternary(0) == 1
    assert real_data._count_to_ternary(1) == 0
    assert real_data._count_to_ternary(2) == 2
    assert real_data._count_to_ternary(56) == 2


def test_parse_tsv_drops_total_column():
    """The ``Total`` column must be dropped; orthogroup IDs become row labels."""
    tsv = (
        "\tSp1\tSp2\tSp3\tTotal\n"
        "OG0000000\t3\t0\t1\t4\n"
        "OG0000001\t0\t0\t0\t0\n"
        "OG0000002\t5\t2\t1\t8\n"
    )
    matrix, row_labels, col_labels = real_data._parse_tsv(tsv)
    assert col_labels == ["Sp1", "Sp2", "Sp3"]
    # 3/0/1 -> 2/1/0 ; 0/0/0 -> 1/1/1 ; 5/2/1 -> 2/2/0
    assert matrix == [
        [2, 1, 0],
        [1, 1, 1],
        [2, 2, 0],
    ]


def test_load_real_matrices_offline_returns_empty(tmp_path, monkeypatch):
    """When the cache is empty and the network is down, the loader returns []."""
    monkeypatch.setattr(real_data, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(real_data, "_fetch_csv", lambda dirname: None)
    assert real_data.load_real_matrices() == []


def test_load_large_matrices_offline_returns_empty(tmp_path, monkeypatch):
    """EggNOG tier: cache empty + every fetch fails => empty list."""
    monkeypatch.setattr(real_data, "_CACHE_DIR", tmp_path)
    # All three external fetches return None; the loader must not raise.
    monkeypatch.setattr(real_data, "_fetch_eggnog_csv", lambda *a, **kw: None)
    assert real_data.load_large_matrices() == []


def test_load_all_real_is_concatenation(monkeypatch):
    """``load_all_real`` is ``load_real_matrices`` + ``load_large_matrices``."""
    monkeypatch.setattr(real_data, "load_real_matrices", lambda: [("small", [], [], [])])
    monkeypatch.setattr(real_data, "load_large_matrices", lambda: [("large", [], [], [])])
    assert real_data.load_all_real() == [("small", [], [], []), ("large", [], [], [])]


@pytest.mark.slow
def test_load_real_matrices_smoke():
    """End-to-end: load every small-tier dataset, check ternary values are in {0,1,2}."""
    datasets = real_data.load_real_matrices()
    if not datasets:
        pytest.skip("no real datasets available (offline)")
    for name, matrix, row_labels, col_labels in datasets:
        assert matrix, f"{name} produced empty matrix"
        m = len(matrix)
        n = len(matrix[0])
        assert n == len(col_labels) > 0
        assert m == len(row_labels)
        for r in matrix:
            assert all(c in (0, 1, 2) for c in r), f"{name} has non-ternary row {r}"


@pytest.mark.slow
def test_load_large_matrices_smoke():
    """End-to-end: load every large-tier dataset, check shape and ternary values."""
    datasets = real_data.load_large_matrices()
    if not datasets:
        pytest.skip("no large-tier datasets available (offline)")
    assert datasets, "expected at least one large-tier dataset"
    has_large_n = False
    for name, matrix, row_labels, col_labels in datasets:
        assert matrix, f"{name} produced empty matrix"
        m = len(matrix)
        n = len(matrix[0])
        assert n == len(col_labels) > 0, f"{name}: {n} cols vs {len(col_labels)} labels"
        assert m == len(row_labels), f"{name}: {m} rows vs {len(row_labels)} labels"
        if n >= 10:
            has_large_n = True
        for r in matrix:
            assert all(c in (0, 1, 2) for c in r), f"{name} has non-ternary row {r}"
    assert has_large_n, "expected at least one dataset with n >= 10"


@pytest.mark.slow
def test_load_all_real_combines_tiers():
    """``load_all_real`` returns both tiers; the large tier contributes n >= 10."""
    datasets = real_data.load_all_real()
    if not datasets:
        pytest.skip("no datasets available (offline)")
    names = [d[0] for d in datasets]
    # Small tier names are mycoplasma_*; large tier names are eggnog_*.
    has_small = any(n.startswith("mycoplasma_") for n in names)
    assert has_small, f"missing small tier in {names}"
    if real_data._EGGNOG_DATASETS:
        has_large = any(n.startswith("eggnog_") for n in names)
        assert has_large, f"missing large tier in {names}"
