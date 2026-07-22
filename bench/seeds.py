"""Canonical seed derivation for MinGap benchmarks.

Single source of truth — do not duplicate the formula elsewhere.

The bench scripts (bench/bench_paper.py and any future tool) must import
`seed_for` from here rather than re-deriving the multipliers inline.
"""

from __future__ import annotations

from typing import Final

_SEED_MUL_TRIAL: Final[int] = 7919
_SEED_MUL_N: Final[int] = 131
_SEED_MUL_K: Final[int] = 17


def seed_for(trial: int, n: int, k: int = 0, gen_offset: int = 0) -> int:
    """Return a deterministic seed for the given (trial, n, k, gen_offset) tuple.

    Multipliers are primes chosen so adjacent parameter values produce seeds
    that are well-separated (no collisions across the (trial, n, k) grid).

    Parameters
    ----------
    trial : int
        Trial index within a given (n, k) configuration.
    n : int
        Number of columns / species in the matrix.
    k : int
        Row-multiplier (k * n = number of rows). Pass 0 if not applicable.
    gen_offset : int
        Disambiguator for the matrix generator (e.g. 0 = random, 1 = satisfiable).
        Pass 0 if not applicable.
    """
    return (
        trial * _SEED_MUL_TRIAL
        + n * _SEED_MUL_N
        + k * _SEED_MUL_K
        + gen_offset
    )