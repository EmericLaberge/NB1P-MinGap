"""Read a syntesim log, build a ternary matrix, run NB1P/MinGap.

Usage:
    python scripts/syntesim_analyze.py <log>            # print matrix + nb1p verdict
    python scripts/syntesim_analyze.py <log> --solve    # also compute a min-gap permutation
    python scripts/syntesim_analyze.py <log> --json     # emit the raw matrix as JSON
"""

from __future__ import annotations

import argparse
import json
import sys

import mingap
from syntesim_bridge import load as syntesim_load


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="syntesim-analyze",
        description="Read a syntesim log, build a ternary matrix, run NB1P/MinGap.",
    )
    p.add_argument("log", help="path to a syntesim JSON-lines log")
    p.add_argument(
        "--solve",
        action="store_true",
        help="also compute a MinGap min-gap permutation",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit the (matrix, leaves, genes) triple as JSON on stdout",
    )
    p.add_argument(
        "--solver",
        default="auto",
        help="backend for mingap (default: auto)",
    )
    return p


def run(args: argparse.Namespace) -> int:
    result = syntesim_load(args.log)

    if args.json:
        json.dump(
            {"matrix": result.matrix, "leaves": result.leaves, "genes": result.genes},
            sys.stdout,
        )
        sys.stdout.write("\n")
        return 0

    m, leaves, genes = result.matrix, result.leaves, result.genes
    n0 = sum(c == 0 for row in m for c in row)
    n1 = sum(c == 1 for row in m for c in row)
    n2 = sum(c == 2 for row in m for c in row)
    print(f"Matrix: {len(leaves)} leaves × {len(genes)} genes")
    print(f"  0 (present): {n0}    1 (lost at edge): {n1}    2 (absent): {n2}")
    print(f"  is_nb1p: {result.is_nb1p()}")
    if args.solve:
        perm, cost = mingap.solve(m, solver=args.solver)
        print(f"  MinGap cost: {cost}")
        print(f"  permutation: {perm}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())