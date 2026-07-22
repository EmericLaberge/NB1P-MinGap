#!/usr/bin/env python3
"""P2-4: registered ILP (Gurobi) solver on the MaxSAT frontier cells.

Random instances, n in {10, 12}, m = 2n, same 5 seeds and 3 s budget as
bench/bench_paper.py maxsat_scaling. Results appended to
bench/bench_paper.json under key ``ilp_frontier``.
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "bench"))

from benchmark_sat import random_matrix  # noqa: E402
from seeds import seed_for  # noqa: E402

OUT = ROOT / "bench" / "bench_paper.json"
TIMEOUT = 3.0


def _ilp_one(m, n, seed, q):
    from mingap import get

    mat = random_matrix(m, n, seed)
    t0 = time.perf_counter()
    r = get("ilp").solve(mat)
    q.put((time.perf_counter() - t0, r.cost))


def main() -> None:
    ctx = mp.get_context("spawn")
    out = json.loads(OUT.read_text())
    rows = list(out.get("ilp_frontier", []))
    existing = {(r["n"], r["m"], r["seed"]) for r in rows}
    for n in (10, 12):
        m = 2 * n
        for s in range(5):
            seed = seed_for(s, n, 2, 0)
            if (n, m, seed) in existing:
                continue
            q = ctx.Queue()
            p = ctx.Process(target=_ilp_one, args=(m, n, seed, q))
            p.start()
            p.join(TIMEOUT)
            row = dict(n=n, m=m, seed=seed, timeout=False, cost=None, ms=None)
            if p.is_alive():
                p.terminate()
                p.join()
                row["timeout"] = True
            else:
                try:
                    dt, cost = q.get_nowait()
                    row.update(cost=cost, ms=dt * 1000)
                except Exception:
                    row.update(timeout=True, error=f"exitcode={p.exitcode}")
            rows.append(row)
            print(f"[ilp] n={n} m={m} seed={seed} -> {row}", flush=True)
            out["ilp_frontier"] = rows
            OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
