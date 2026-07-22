"""NB1P CLI — solve / check."""
from __future__ import annotations

import argparse
import json
import sys

import nb1p


def _parse_matrix(s: str) -> list[list[int]]:
    """Parse a JSON matrix literal like ``"[[0,1,2],[1,0,2]]"`` into a nested list of ints."""
    return json.loads(s)


def run(args: argparse.Namespace) -> int:
    """Dispatch CLI: solve, enumerate (``--all``), or decision-only check (``--check-only``)."""
    matrix = _parse_matrix(args.matrix)
    if args.all:
        perms = nb1p.solve_all(matrix, solver=args.solver)
        print(json.dumps(perms))
        return 0
    if args.check_only:
        print(nb1p.check(matrix))
        return 0
    result = nb1p.solve(matrix, solver=args.solver)
    print(json.dumps({"satisfiable": result.satisfiable, "permutation": result.permutation}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``nb1p`` CLI argument parser (subcommands-free; flags only)."""
    p = argparse.ArgumentParser(prog="nb1p", description="NB1P decision solver")
    p.add_argument("matrix", help="JSON ternary matrix")
    p.add_argument("--solver", default="auto")
    p.add_argument("--all", action="store_true", help="enumerate all witnesses")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse argv, dispatch to ``run``, return its exit code."""
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
