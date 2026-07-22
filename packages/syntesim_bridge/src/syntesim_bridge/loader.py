"""syntesim_bridge — load syntesim simulation logs into MinGap solvers.

A thin layer over `syntesim.from_log` that returns nb1p / mingap-shaped
objects so the rest of MinGap can consume a syntesim output without
knowing about syntesim's data model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class BridgeResult:
    """A ternary matrix + labels, ready for nb1p.check / mingap.solve."""

    matrix: list[list[int]]
    leaves: list[str]
    genes: list[str]

    def is_nb1p(self) -> bool:
        import nb1p

        return nb1p.check(self.matrix)


def load(log_source: str | Iterable[str]) -> BridgeResult:
    """Read a syntesim log (path or iterable of lines) and return the
    (matrix, leaves, genes) triple plus an `is_nb1p` shortcut."""
    from syntesim import from_log

    matrix, leaves, genes = from_log(log_source)
    return BridgeResult(matrix=matrix, leaves=leaves, genes=genes)


__all__ = ["BridgeResult", "load"]
