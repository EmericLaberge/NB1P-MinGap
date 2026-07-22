"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

import nb1p
from mingap import _registry as mingap_registry
from nb1p import _registry as nb1p_registry


@pytest.fixture
def paper_matrix() -> nb1p.Matrix:
    """6×5 ternary matrix from the NB1P paper (NB1P-satisfiable)."""
    return nb1p.paper_example()


@pytest.fixture
def satisfiable_matrix() -> nb1p.Matrix:
    """A small NB1P-satisfiable matrix (identity already works)."""
    return [
        [1, 1, 0],
        [0, 1, 1],
    ]


@pytest.fixture
def unsatisfiable_matrix() -> nb1p.Matrix:
    """The non-betweenness gadget on 3 columns — NB1P-unsatisfiable.

    Two rows force column 1 both inside and outside the 0/2 span; no
    permutation satisfies both.
    """
    return [
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ]


def _loadable(registry, name: str) -> bool:
    return name in registry.available()


def _functional(registry, name: str) -> bool:
    try:
        registry.get(name).solve([[1, 0, 1]])
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


@pytest.fixture(params=["brute_force", "sat"])
def nb1p_solver(request) -> str:
    """Parametrised NB1P backend, skipping unavailable ones."""
    name = request.param
    if not _loadable(nb1p_registry, name):
        pytest.skip(f"NB1P backend '{name}' not installed")
    if not _functional(nb1p_registry, name):
        pytest.skip(f"NB1P backend '{name}' not implemented")
    return name


@pytest.fixture(params=["brute_force", "maxsat", "greedy"])
def mingap_solver(request) -> str:
    """Parametrised MinGap backend, skipping unavailable ones."""
    name = request.param
    if not _loadable(mingap_registry, name):
        pytest.skip(f"MinGap backend '{name}' not installed")
    if not _functional(mingap_registry, name):
        pytest.skip(f"MinGap backend '{name}' not implemented")
    return name


