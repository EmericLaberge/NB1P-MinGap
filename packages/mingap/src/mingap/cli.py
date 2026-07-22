"""MinGap CLI — ``solve`` (NB1P/MinGap) and ``analyze`` (pipeline) subcommands."""
from __future__ import annotations

import argparse
import ast
import sys

import mingap
import nb1p


def _parse_matrix(text: str) -> list[list[int]]:
    """Parse a matrix literal like ``[[0,1,2],[1,0,2]]``."""
    value = ast.literal_eval(text)
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(r, (list, tuple)) for r in value
    ):
        raise ValueError("matrix must be a list of rows")
    return [list(r) for r in value]


def _run_solve(args: argparse.Namespace) -> int:
    """Solve NB1P (default) or MinGap (``--problem mingap``) for a single matrix."""
    matrix = _parse_matrix(args.matrix)
    if args.problem == "nb1p":
        if args.all:
            perms = nb1p.solve_all(matrix, solver=args.solver)
            print(f"{len(perms)} valid permutation(s):")
            for p in perms:
                print(f"  {p}")
        else:
            result = nb1p.solve(matrix, solver=args.solver)
            if result.satisfiable:
                print(f"SAT — permutation: {result.permutation}")
            else:
                print("UNSAT")
                return 1
    else:  # mingap
        result = mingap.solve(matrix, solver=args.solver)
        print(f"min gaps = {result.cost} — permutation: {result.permutation}")
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    """Check NB1P first; if SAT, run MinGap optimisation on the same matrix."""
    matrix = _parse_matrix(args.matrix)
    report = mingap.analyze(matrix, solver=args.solver)
    if report.is_nb1p:
        print(f"NB1P ✓ — gap-free permutation: {report.permutation}")
    else:
        print(f"not NB1P — min gaps = {report.n_gaps} "
              f"— permutation: {report.permutation}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``mingap`` CLI: ``solve`` and ``analyze`` subcommands share one parser."""
    p = argparse.ArgumentParser(prog="mingap-cli", description="MinGap solvers")
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("solve", help="solve NB1P or MinGap for a matrix")
    ps.add_argument("matrix", help="matrix literal, e.g. '[[0,1,2],[1,0,2]]'")
    ps.add_argument("--problem", choices=["nb1p", "mingap"], default="nb1p",
                    help="which problem to solve (default: nb1p)")
    ps.add_argument("--solver", default="auto",
                    help="backend name or 'auto' (default: auto)")
    ps.add_argument("--all", action="store_true",
                    help="enumerate all solutions (nb1p only)")
    ps.set_defaults(func=_run_solve)

    pa = sub.add_parser("analyze", help="check NB1P and minimise gaps")
    pa.add_argument("matrix", help="matrix literal, e.g. '[[1,0,1],[0,1,1]]'")
    pa.add_argument("--solver", default="auto",
                    help="backend name or 'auto' (default: auto)")
    pa.set_defaults(func=_run_analyze)

    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse argv and dispatch to the selected subcommand."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
