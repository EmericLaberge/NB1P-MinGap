"""syntesim_bridge — load syntesim simulation logs into MinGap's matrix solvers."""

from .loader import BridgeResult, load

__all__ = [
    "BridgeResult",
    "load",
]
