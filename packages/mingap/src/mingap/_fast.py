"""Numba-accelerated block-cost kernels for exact MinGap search.

Optional dependency: ``pip install mingap[sa]`` (numpy + numba).
"""

from __future__ import annotations

import os

import numpy as np
from numba import njit, prange

from mingap.types import Matrix, Permutation

# factorials 0! … 12! — enough for n ≤ 12 brute force
_FACT = np.array(
    [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600],
    dtype=np.int64,
)

# Use parallel enumeration from n ≥ 9 (9! = 362880).
_PARALLEL_MIN_N = 9


@njit(cache=True)
def block_cost_njit(A: np.ndarray, perm: np.ndarray, m: int, n: int) -> int:
    """Block cost on a contiguous int8 matrix and int64 permutation."""
    total = 0
    for i in range(m):
        in_block = False
        has_one = False
        blocks = 0
        for k in range(n):
            v = A[i, perm[k]]
            if v == 0:
                if in_block and has_one:
                    blocks += 1
                in_block = False
                has_one = False
            else:
                in_block = True
                if v == 1:
                    has_one = True
        if in_block and has_one:
            blocks += 1
        if blocks > 1:
            total += blocks - 1
    return total


@njit(cache=True)
def _unrank_perm(rank: int, n: int, perm: np.ndarray) -> None:
    """Write Lehmer-unranked permutation into *perm* (length *n*)."""
    pool = np.empty(n, dtype=np.int64)
    for i in range(n):
        pool[i] = i
    r = rank
    for i in range(n):
        fact = _FACT[n - 1 - i]
        j = r // fact
        r = r % fact
        perm[i] = pool[j]
        for k in range(j, n - 1 - i):
            pool[k] = pool[k + 1]


@njit(cache=True)
def brute_force_njit(A: np.ndarray, m: int, n: int) -> tuple[np.ndarray, int]:
    """Exhaustive MinGap: try all ``n!`` permutations, return best perm + cost."""
    perm = np.arange(n, dtype=np.int64)
    c = np.zeros(n, dtype=np.int64)
    best_score = m * n + 1
    best_perm = perm.copy()

    i = 0
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                perm[0], perm[i] = perm[i], perm[0]
            else:
                perm[c[i]], perm[i] = perm[i], perm[c[i]]
            s = block_cost_njit(A, perm, m, n)
            if s < best_score:
                best_score = s
                for j in range(n):
                    best_perm[j] = perm[j]
                if s == 0:
                    break
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1
    return best_perm, best_score


@njit(parallel=True, cache=True)
def brute_force_parallel_njit(
    A: np.ndarray, m: int, n: int, n_workers: int
) -> tuple[np.ndarray, int]:
    """Parallel exhaustive MinGap: partition ``n!`` ranks across workers."""
    nfact = _FACT[n]
    chunk = (nfact + n_workers - 1) // n_workers
    local_best = np.full(n_workers, m * n + 1, dtype=np.int32)
    local_rank = np.zeros(n_workers, dtype=np.int64)

    for wid in prange(n_workers):
        perm = np.empty(n, dtype=np.int64)
        start = wid * chunk
        end = start + chunk
        if end > nfact:
            end = nfact
        if start >= nfact:
            continue
        lb = m * n + 1
        lr = np.int64(0)
        for rank in range(start, end):
            _unrank_perm(rank, n, perm)
            s = block_cost_njit(A, perm, m, n)
            if s < lb:
                lb = s
                lr = rank
                if s == 0:
                    break
        local_best[wid] = lb
        local_rank[wid] = lr

    best_score = m * n + 1
    best_rank = np.int64(0)
    for wid in range(n_workers):
        if local_best[wid] < best_score:
            best_score = local_best[wid]
            best_rank = local_rank[wid]

    best_perm = np.empty(n, dtype=np.int64)
    _unrank_perm(best_rank, n, best_perm)
    return best_perm, best_score


@njit(parallel=True, cache=True)
def block_cost_batch_njit(
    A: np.ndarray, ranks: np.ndarray, m: int, n: int
) -> np.ndarray:
    """Evaluate block cost for a batch of Lehmer-ranked permutations."""
    batch = ranks.shape[0]
    out = np.empty(batch, dtype=np.int32)
    for bi in prange(batch):
        perm = np.empty(n, dtype=np.int64)
        _unrank_perm(int(ranks[bi]), n, perm)
        out[bi] = block_cost_njit(A, perm, m, n)
    return out


def default_workers() -> int:
    """Thread count for parallel brute force (override via ``MININV_WORKERS``)."""
    env = os.environ.get("MININV_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return os.cpu_count() or 1


def brute_force_best(
    A: np.ndarray, m: int, n: int, *, parallel: bool | None = None
) -> tuple[np.ndarray, int]:
    """Pick sequential or parallel brute force depending on *n* and *parallel*."""
    use_parallel = parallel if parallel is not None else n >= _PARALLEL_MIN_N
    if use_parallel and n >= _PARALLEL_MIN_N:
        return brute_force_parallel_njit(A, m, n, default_workers())
    return brute_force_njit(A, m, n)


def matrix_to_array(matrix: Matrix) -> np.ndarray:
    """Convert a ternary matrix to a contiguous int8 numpy array."""
    return np.array(matrix, dtype=np.int8)


def permutation_to_array(perm: Permutation) -> np.ndarray:
    """Copy a permutation list to int64 numpy array."""
    return np.array(perm, dtype=np.int64)


__all__ = [
    "block_cost_batch_njit",
    "block_cost_njit",
    "brute_force_best",
    "brute_force_njit",
    "brute_force_parallel_njit",
    "default_workers",
    "matrix_to_array",
    "permutation_to_array",
]
