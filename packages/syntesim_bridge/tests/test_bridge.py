"""Tests for syntesim_bridge — feeding syntesim logs into MinGap solvers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import syntesim_bridge as sb


# ── hand-crafted log fixtures ────────────────────────────────────────────


SYNTHETIC_LOG = [
    json.dumps({"state": {"S1": {"X1": [["G1", True], ["G2", False]]}}}),
    json.dumps(
        {"event": "Speciation", "parent_id": "S1", "child1_id": "S2", "child2_id": "S3"}
    ),
    json.dumps(
        {
            "state": {
                "S1": {"X1": [["G1", True], ["G2", False]]},
                "S2": {"X1": [["G1", True], ["G2", False]]},
                "S3": {"X1": [["G1", True], ["G2", False]]},
            }
        }
    ),
    json.dumps(
        {
            "event": "Loss",
            "species_id": "S2",
            "synteny_id": "X1",
            "start": "0",
            "end": "0",
        }
    ),
    json.dumps(
        {
            "state": {
                "S1": {"X1": [["G1", True], ["G2", False]]},
                "S2": {"X1": [["G2", False]]},
                "S3": {"X1": [["G1", True], ["G2", False]]},
            }
        }
    ),
]


def test_hand_crafted_log_parses():
    result = sb.load(SYNTHETIC_LOG)
    assert sorted(result.leaves) == ["S1", "S2", "S3"]
    assert sorted(result.genes) == ["G1", "G2"]
    assert len(result.matrix) == 3
    assert all(len(row) == 2 for row in result.matrix)


def test_loss_event_produces_one_cell():
    result = sb.load(SYNTHETIC_LOG)
    s1_idx = result.leaves.index("S1")
    s2_idx = result.leaves.index("S2")
    s3_idx = result.leaves.index("S3")
    g1_idx = result.genes.index("G1")
    g2_idx = result.genes.index("G2")
    assert result.matrix[s1_idx][g1_idx] == 0
    assert result.matrix[s1_idx][g2_idx] == 0
    assert result.matrix[s2_idx][g1_idx] == 1
    assert result.matrix[s2_idx][g2_idx] == 0
    assert result.matrix[s3_idx][g1_idx] == 0
    assert result.matrix[s3_idx][g2_idx] == 0


def test_matrix_feeds_nb1p():
    import nb1p

    result = sb.load(SYNTHETIC_LOG)
    assert nb1p.check(result.matrix) is True
    assert result.is_nb1p() is True


def test_other_events_ignored():
    log = list(SYNTHETIC_LOG) + [
        json.dumps({"event": "Transfer", "src": "S1", "target": "S3", "gene": "G1"}),
        json.dumps({"event": "Duplication", "species": "S3", "gene": "G1"}),
        json.dumps(
            {
                "event": "Cut",
                "species_id": "S3",
                "synteny_id": "X1",
                "position": "1",
                "child1_id": "X2",
                "child2_id": "X3",
            }
        ),
        json.dumps({"event": "Extinction", "species_id": "S1"}),
    ]
    result = sb.load(log)
    assert sorted(result.leaves) == ["S1", "S2", "S3"]


# ── end-to-end with the real simulate script ─────────────────────────────


def _have_real_simulation() -> str | None:
    """Pipe `syntesim/simulate` into a tempfile; return its path or None."""
    syntesim_root = Path("/home/emeric/maitrise/redaction/syntesim-mirror")
    if not (syntesim_root / "simulate").exists():
        return None
    fd, log_path = tempfile.mkstemp(prefix="syntesim_", suffix=".log")
    try:
        with os.fdopen(fd, "w") as fh:
            proc = subprocess.run(
                [sys.executable, "simulate"],
                cwd=syntesim_root,
                stdout=fh,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=30,
            )
        if proc.returncode != 0:
            os.unlink(log_path)
            return None
        return log_path
    except Exception:
        if os.path.exists(log_path):
            os.unlink(log_path)
        raise


def test_real_simulation_to_matrix():
    log_path = _have_real_simulation()
    if log_path is None:
        pytest.skip("syntesim/simulate not available")
    try:
        result = sb.load(log_path)
        assert len(result.matrix) > 0
        assert len(result.genes) > 0
        flat = [c for row in result.matrix for c in row]
        assert 0 in flat
        assert 2 in flat
        assert all(c in (0, 1, 2) for c in flat)
    finally:
        if os.path.exists(log_path):
            os.unlink(log_path)
