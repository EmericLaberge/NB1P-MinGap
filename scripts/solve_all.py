"""Solve or re-solve scenarios in the DuckDB database.

By default solves unsolved rows (mingap IS NULL).
Use --all to re-solve everything, or --solved to re-solve only already-solved rows.

Usage:
    python scripts/solve_all.py data/scenarios.duckdb                   # solve unsolved only
    python scripts/solve_all.py data/scenarios.duckdb --all             # re-solve everything
    python scripts/solve_all.py data/scenarios.duckdb --solved          # re-solve already solved
    python scripts/solve_all.py data/scenarios.duckdb --limit 5000
    python scripts/solve_all.py data/scenarios_sglpp.duckdb --sample-large 0.1  # 10% of large matrices
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb

from mingap import solve as mg_solve
from nb1p import solve as nb1p_solve


def _table_name(db_path: str) -> str:
    """Derive table name from the DB file's stem (e.g. scenarios_sglpp.duckdb -> scenarios_sglpp)."""
    return Path(db_path).stem


def _build_where(mode: str, tbl: str, con: duckdb.DuckDBPyConnection, sample_large: float, max_genes: int | None) -> tuple[str, int]:
    """Build WHERE clause with optional hard cutoff and sampling for large matrices."""
    base = {
        "unsolved": "mingap IS NULL",
        "solved": "mingap IS NOT NULL",
        "all": "1=1",
    }[mode]

    cols = {r[0] for r in con.execute(
        f"SELECT column_name FROM information_schema.columns WHERE table_name = '{tbl}'"
    ).fetchall()}
    has_size = "n_genes" in cols and "n_leaves" in cols

    if not has_size:
        count = con.execute(f"SELECT COUNT(*) FROM {tbl} WHERE {base}").fetchone()[0]
        return base, count

    parts: list[str] = [base]

    # Hard cutoff: skip matrices with too many genes entirely
    if max_genes is not None:
        parts.append(f"n_genes <= {max_genes}")

    # Sampling: for matrices above P75 of n_genes, keep only 1/N
    if sample_large < 1.0:
        threshold = con.execute(
            f"SELECT CAST(percentile_cont(0.75) WITHIN GROUP (ORDER BY n_genes) AS INTEGER) FROM {tbl} WHERE {base}"
        ).fetchone()[0]
        denom = max(1, round(1.0 / sample_large))
        # sample only applies between threshold and max_genes
        parts.append(f"(n_genes <= {threshold} OR (id % {denom} = 0))")
        print(f"  Sampling 1/{denom} of matrices with n_genes > {threshold}")

    where = " AND ".join(f"({p})" for p in parts)
    count = con.execute(f"SELECT COUNT(*) FROM {tbl} WHERE {where}").fetchone()[0]
    total = con.execute(f"SELECT COUNT(*) FROM {tbl} WHERE {base}").fetchone()[0]
    skipped = total - count
    if skipped:
        print(f"  Keeping {count:,} of {total:,} (skipped {skipped:,})")
    return where, count


def solve_all(db_path: str, *, batch_size: int = 500, limit: int | None = None, mode: str = "unsolved", sample_large: float = 1.0, max_genes: int | None = None) -> None:
    con = duckdb.connect(db_path)
    tbl = _table_name(db_path)
    # Ensure the expected columns exist (SGLPP tables may not have them)
    cols = {r[0] for r in con.execute(
        f"SELECT column_name FROM information_schema.columns WHERE table_name = '{tbl}'"
    ).fetchall()}
    if "nb1p" not in cols:
        con.execute(f"ALTER TABLE {tbl} ADD COLUMN nb1p BOOLEAN")
    if "mingap" not in cols:
        con.execute(f"ALTER TABLE {tbl} ADD COLUMN mingap INTEGER")

    where, count = _build_where(mode, tbl, con, sample_large, max_genes)
    print(f"  {count:,} {mode} scenarios in {db_path}")

    to_solve = min(count, limit) if limit else count
    print(f"  Solving {to_solve:,} scenarios (batch_size={batch_size}, mode={mode})")

    done = 0
    errors = 0
    t0 = time.time()

    while done < to_solve:
        batch = min(batch_size, to_solve - done)
        rows = con.execute(
            f"SELECT id, matrix FROM {tbl} WHERE {where} ORDER BY id LIMIT ?",
            [batch],
        ).fetchall()

        if not rows:
            break

        updates: list[tuple] = []
        for row_id, matrix_json in rows:
            mat_raw = json.loads(matrix_json) if matrix_json else []
            nb1p_val: bool | None = None
            mingap_val: int | None = None

            if not mat_raw:
                nb1p_val = True
                mingap_val = 0
            else:
                try:
                    res = nb1p_solve(mat_raw)
                    nb1p_val = res.satisfiable
                except Exception:
                    errors += 1
                try:
                    res = mg_solve(mat_raw)
                    mingap_val = res.cost
                except Exception:
                    errors += 1

            updates.append((nb1p_val, mingap_val, row_id))

        con.executemany(
            f"UPDATE {tbl} SET nb1p = ?, mingap = ? WHERE id = ?",
            updates,
        )

        done += len(updates)
        elapsed = time.time() - t0
        rate = done / max(elapsed, 0.001)
        print(f"    {done:>8,}/{to_solve:,}  ({rate:.0f}/s, {errors} errors)")

    elapsed = time.time() - t0
    remaining_unsolved = con.execute(
        f"SELECT COUNT(*) FROM {tbl} WHERE mingap IS NULL"
    ).fetchone()[0]
    total_solved = con.execute(
        f"SELECT COUNT(*) FROM {tbl} WHERE mingap IS NOT NULL"
    ).fetchone()[0]

    print(f"\n  Done in {elapsed:.1f}s: {done:,} re-solved, {errors} errors")
    print(f"  {remaining_unsolved:,} unsolved, {total_solved:,} total solved")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve scenarios in DuckDB")
    ap.add_argument("db", help="Path to scenarios.duckdb")
    ap.add_argument("--batch-size", type=int, default=500, help="Rows per batch")
    ap.add_argument("--limit", type=int, default=None, help="Max rows to solve")

    ap.add_argument("--sample-large", type=float, default=1.0,
                    help="Fraction of large matrices to solve, e.g. 0.1 keeps 10%% (default: 1.0 = all)")
    ap.add_argument("--max-genes", type=int, default=None,
                    help="Hard cutoff: skip matrices with more genes than this")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Solve all rows")
    mode.add_argument("--solved", action="store_true", help="Re-solve already solved rows")
    args = ap.parse_args()
    m = "all" if args.all else ("solved" if args.solved else "unsolved")
    solve_all(args.db, batch_size=args.batch_size, limit=args.limit, mode=m,
              sample_large=args.sample_large, max_genes=args.max_genes)


if __name__ == "__main__":
    main()
