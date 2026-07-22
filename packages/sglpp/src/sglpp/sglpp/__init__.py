"""SGLPP — Small Gain-Loss Parsimony Problem.
Dollo-consistent gene evolution where each gene family is gained once
(anywhere in the tree) and may then be lost independently on descendant
branches. Linear-time DP from Gascon, Delabrouk, El-Mabrouk (RECOMB-CG
2026, LNBI 16569), ported from https://github.com/UdeM-LBIT/InOutParsimony.
"""

from __future__ import annotations

from sglpp.sglpp.algorithm import DEFAULT_DELTA_GAIN, DEFAULT_DELTA_LOSS, solve
from sglpp.sglpp.result import SGLPPResult

__all__ = [
    "DEFAULT_DELTA_GAIN",
    "DEFAULT_DELTA_LOSS",
    "SGLPPResult",
    "solve",
]
