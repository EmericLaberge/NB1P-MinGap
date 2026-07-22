#!/usr/bin/env python3
"""Full experiment suite for the paper.

Parts:
  sat      — SAT scaling, full grid n x k, both generators, SAT ratios
  maxsat   — MaxSAT (RC2) MinGap scaling with per-instance pebble timeouts
  fuzz     — MaxSAT vs brute-force cost agreement on small n (n=3..8)
  fuzz10   — MaxSAT vs brute-force cost agreement on small n (n=3..10)
  syntesim — simulated phylogenetic instances via syntesim
  bounds   — LP lower bounds and feasible upper bounds on generated + real data
  tsp_eggnog — tsp_approx (Christofides/CBM) vs greedy/lp_round on EggNOG large data
  tsp_exact — tsp_approx/greedy/greedy_rollout/lp_round vs exact optimum (small n)
  real     — exact/bounded solving on real biological matrices

Usage: uv run python bench/bench_paper.py {sat|maxsat|fuzz|fuzz10|syntesim|bounds|tsp_eggnog|tsp_exact|real|all}
Results are written incrementally to bench/bench_paper.json

Fuzz range extended to n=3..8; extended n=3..10 lives under `fuzz_n3_n10`.
See also review §P3 (was 3..7).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_sat import benchmark, random_matrix, sat_matrix  # noqa: E402

from seeds import seed_for  # noqa: E402

OUT = Path(__file__).with_suffix(".json")


def save(out: dict) -> None:
    OUT.write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------- SAT scaling
def sat_scaling(out: dict) -> None:
    sizes = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    res = out.get("sat_scaling", {})
    for gen_name, gen in (("random", random_matrix), ("satisfiable", sat_matrix)):
        for n in sizes:
            for k in (1, 2, 5):
                m = k * n
                key = f"{gen_name}|n={n}|m={m}"
                if key in res:
                    continue
                trials = 1000 if n <= 30 else 500
                builds, solves, clauses = [], [], []
                sat_yes = 0
                for t in range(trials):
                    gen_offset = 0 if gen_name == "random" else 1
                    seed = seed_for(t, n, k, gen_offset)
                    r = benchmark(gen(m, n, seed=seed))
                    builds.append(r["t_build_ms"])
                    solves.append(r["t_solve_ms"])
                    clauses.append(r["n_clauses"])
                    sat_yes += bool(r["sat"])
                res[key] = dict(
                    n=n, m=m, trials=trials,
                    clauses=statistics.mean(clauses),
                    build=statistics.mean(builds), build_sd=statistics.stdev(builds),
                    solve=statistics.mean(solves), solve_sd=statistics.stdev(solves),
                    sat_pct=100.0 * sat_yes / trials,
                )
                print(
                    f"[sat] {key} clauses={res[key]['clauses']:.0f} "
                    f"total={res[key]['build'] + res[key]['solve']:.1f}ms "
                    f"sat%={res[key]['sat_pct']:.1f}",
                    flush=True,
                )
                out["sat_scaling"] = res
                save(out)


# ------------------------------------------------------------- MaxSAT scaling
def _maxsat_one(payload):
    """Worker: solve one MinGap instance with RC2, return (seconds, cost)."""
    import time

    from mingap import get

    gen_name, m, n, seed = payload
    gen = random_matrix if gen_name == "random" else sat_matrix
    mat = gen(m, n, seed)
    t0 = time.perf_counter()
    r = get("maxsat").solve(mat)
    return time.perf_counter() - t0, r.cost


def maxsat_scaling(out: dict) -> None:
    from concurrent.futures import TimeoutError

    from pebble import ProcessPool

    sizes = [5, 8, 10, 12, 15, 20, 25, 30]
    timeout = 3.0
    n_trials = 5
    res = out.get("maxsat_scaling", {})
    with ProcessPool(max_workers=8) as pool:
        for gen_name in ("random", "satisfiable"):
            for n in sizes:
                for k in (1, 2):
                    m = k * n
                    key = f"{gen_name}|n={n}|m={m}"
                    if key in res:
                        continue
                    times, costs = [], []
                    n_timeout = 0
                    consec_timeout = 0
                    trials_done = 0
                    seeds = list(range(n_trials))
                    for wave_start in range(0, len(seeds), 8):
                        wave = seeds[wave_start : wave_start + 8]
                        gen_offset = 0 if gen_name == "random" else 1
                        futs = [
                            pool.schedule(
                                _maxsat_one,
                                args=[(gen_name, m, n, seed_for(s, n, k, gen_offset))],
                                timeout=timeout,
                            )
                            for s in wave
                        ]
                        for f in futs:
                            trials_done += 1
                            try:
                                dt, cost = f.result()
                                times.append(dt * 1000)
                                costs.append(cost)
                                consec_timeout = 0
                            except TimeoutError:
                                n_timeout += 1
                                consec_timeout += 1
                            except Exception as e:  # noqa: BLE001
                                print(f"  worker error: {e!r}", flush=True)
                                n_timeout += 1
                                consec_timeout += 1
                        if consec_timeout >= 3:
                            break
                    key = f"{gen_name}|n={n}|m={m}"
                    solved = trials_done - n_timeout
                    res[key] = dict(
                        n=n, m=m, trials=trials_done, timeouts=n_timeout,
                        solved_pct=100.0 * solved / trials_done if trials_done else 0.0,
                        t_mean=statistics.mean(times) if times else None,
                        t_median=statistics.median(times) if times else None,
                        t_max=max(times) if times else None,
                        cost_mean=statistics.mean(costs) if costs else None,
                    )
                    print(
                        f"[maxsat] {key} solved={res[key]['solved_pct']:.0f}% "
                        f"median={res[key]['t_median']}ms max={res[key]['t_max']}ms",
                        flush=True,
                    )
                    out["maxsat_scaling"] = res
                    save(out)

def fuzz_validate(
    out: dict,
    *,
    n_max: int = 9,
    key: str = "fuzz",
    trials_per_n: "int | dict[int, int]" = 300,
    note: str = "m=2n, random weights 3:3:4, n in 3..8 (extended from 3..7 per review P3)",
) -> None:
    from mingap import get

    bf, ms = get("brute_force"), get("maxsat")

    def trials_for(n: int) -> int:
        if isinstance(trials_per_n, dict):
            return trials_per_n.get(n, 300)
        return trials_per_n

    mismatch, total, worst = 0, 0, 0
    t0 = time.perf_counter()
    for n in range(3, n_max):  # n=3..n_max-1
        n_trials = trials_for(n)
        for t in range(n_trials):
            mat = random_matrix(2 * n, n, seed=seed_for(t, n))
            c_bf = bf.solve(mat).cost
            c_ms = ms.solve(mat).cost
            total += 1
            if c_bf != c_ms:
                mismatch += 1
                worst = max(worst, abs(c_bf - c_ms))
                if mismatch <= 5:
                    print(f"MISMATCH n={n} t={t} bf={c_bf} ms={c_ms} {mat}", flush=True)
        print(f"[fuzz] n={n} trials={n_trials} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    out[key] = dict(
        trials=total, mismatches=mismatch, max_abs_diff=worst,
        note=note,
    )
    save(out)


def fuzz_validate_n3_n10(out: dict) -> None:
    """Extended fuzz: brute-force vs MaxSAT on n=3..10.

    n=10 is reduced to 50 trials (vs 300 elsewhere) because brute force on
    10 elements enumerates 3628800 permutations per matrix; full 300 trials
    would exceed 30 minutes. Other widths keep 300 trials."""
    fuzz_validate(
        out,
        n_max=11,
        key="fuzz_n3_n10",
        trials_per_n={10: 50},
        note="m=2n, random weights 3:3:4, n in 3..10 (extended from 3..8; was 3..7 in fuzz_n3_n7)",
    )


# --------------------------------------------------------- Syntesim instances
def _solve_matrix(mat):
    """Worker: solve one matrix with RC2, return (seconds, cost)."""
    import time

    from mingap import get

    t0 = time.perf_counter()
    r = get("maxsat").solve(mat)
    return time.perf_counter() - t0, r.cost


def syntesim_instances(out: dict) -> None:
    import numpy as np

    import nb1p
    import nb1p
    from syntesim import Event, Gene, Species, State, Synteny
    from syntesim.bridge import from_log

    rows = []
    for n_leaves, n_genes in ((8, 10), (8, 12), (10, 10), (10, 12), (12, 10), (12, 12)):
        for rep in range(2):
            seed = n_leaves * 1000 + n_genes * 100 + rep
            state = State.of(Species.of(Synteny.of(*[Gene() for _ in range(n_genes)])))
            state.generator = np.random.default_rng(seed)
            buf = StringIO()
            state.log_to(buf)
            while len(state) < n_leaves:
                state.event({Event.Speciation: 1.0})
            weights = {}
            for name, w in (("Loss", 2.0), ("Gain", 0.1), ("Transfer", 0.2), ("Duplication", 0.2), ("Cut", 0.05), ("Join", 0.05)):
                ev = getattr(Event, name, None)
                if ev is not None:
                    weights[ev] = w
            for _ in range(30 + 6 * rep + 2 * n_genes):
                state.event(weights)
            mat, _, _ = from_log(buf.getvalue().splitlines())
            from concurrent.futures import TimeoutError

            from pebble import ProcessPool

            with ProcessPool(max_workers=1) as pool:
                fut = pool.schedule(_solve_matrix, args=[mat], timeout=5.0)
                try:
                    dt, cost_val = fut.result()
                    t_ms = dt * 1000
                except TimeoutError:
                    t_ms, cost_val = None, None
            rows.append(dict(
                seed=seed, leaves=n_leaves, genes0=n_genes,
                m=len(mat), n=len(mat[0]),
                nb1p=bool(nb1p.check(mat)), cost=cost_val, t_ms=t_ms,
            ))
            print(
                f"[syntesim] L={n_leaves} G={n_genes} rep={rep} "
                f"m={len(mat)} n={len(mat[0])} nb1p={rows[-1]['nb1p']} cost={cost_val}",
                flush=True,
            )
            out["syntesim"] = rows
            save(out)
# -------------------------------------------------------------------- Bounds
def bounds_bench(out: dict) -> None:
    """Record certified LP lower bounds and feasible solver upper bounds."""
    from concurrent.futures import TimeoutError
    import multiprocessing

    from pebble import ProcessPool

    from mingap import get
    from real_data import load_all_real

    instances = []
    for gen_name, gen in (("random", random_matrix), ("satisfiable", sat_matrix)):
        for n in (5, 8, 10, 12, 15, 20):
            m = 2 * n
            seed = seed_for(0, n, 2, 0 if gen_name == "random" else 1)
            instances.append((f"{gen_name}|n={n}|m={m}", gen(m, n, seed), gen_name, seed))
    for name, matrix, _, _ in load_all_real():
        instances.append((name, matrix, "real", None))

    rows: list[dict] = list(out.get("bounds", []))
    existing_names = {row["name"] for row in rows}
    greedy = get("greedy")
    greedy_rollout = get("greedy_rollout")
    brute_force = get("brute_force")
    try:
        lp_relax = get("lp_relax")
    except ImportError:
        lp_relax = None
    try:
        lp_round = get("lp_round")
    except ImportError:
        lp_round = None

    maxsat_timeout_s = 5.0
    with ProcessPool(max_workers=1, context=multiprocessing.get_context("spawn")) as pool:
        for name, matrix, source, seed in instances:
            if name in existing_names:
                print(f"[bounds] {name}: already in JSON, skipping", flush=True)
                continue

            m = len(matrix)
            n = len(matrix[0]) if m else 0

            lp_bound = None
            lp_ms = None
            if lp_relax is not None:
                t0 = time.perf_counter()
                try:
                    lp_bound = float(lp_relax.solve(matrix).cost)
                except Exception as exc:  # noqa: BLE001
                    print(f"[bounds] LP relaxation unavailable: {exc!r}", flush=True)
                    lp_relax = None
                lp_ms = (time.perf_counter() - t0) * 1000.0

            lp_round_cost = None
            lp_round_ms = None
            if lp_round is not None:
                t0 = time.perf_counter()
                try:
                    lp_round_cost = lp_round.solve(matrix).cost
                except Exception as exc:  # noqa: BLE001
                    print(f"[bounds] LP rounding unavailable: {exc!r}", flush=True)
                    lp_round = None
                lp_round_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            greedy_cost = greedy.solve(matrix).cost
            greedy_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            greedy_rollout_cost = greedy_rollout.solve(matrix).cost
            greedy_rollout_ms = (time.perf_counter() - t0) * 1000.0

            brute_force_cost = None
            brute_force_ms = None
            if n <= 10:
                t0 = time.perf_counter()
                brute_force_cost = brute_force.solve(matrix).cost
                brute_force_ms = (time.perf_counter() - t0) * 1000.0

            maxsat_cost = None
            maxsat_ms = None
            maxsat_timeout = False
            maxsat_error = None
            future = pool.schedule(_solve_matrix, args=[matrix], timeout=maxsat_timeout_s)
            try:
                maxsat_seconds, maxsat_cost = future.result()
                maxsat_ms = maxsat_seconds * 1000.0
            except TimeoutError:
                maxsat_timeout = True
            except Exception as exc:  # noqa: BLE001
                maxsat_error = repr(exc)

            row: dict[str, object] = dict(
                name=name, source=source, seed=seed, m=m, n=n,
                lp_relax_lb=lp_bound, lp_relax_ms=lp_ms,
                lp_round_ub=lp_round_cost, lp_round_ms=lp_round_ms,
                greedy_ub=greedy_cost, greedy_ms=greedy_ms,
                greedy_rollout_ub=greedy_rollout_cost,
                greedy_rollout_ms=greedy_rollout_ms,
                maxsat_ub=maxsat_cost, maxsat_ms=maxsat_ms,
                maxsat_timeout=maxsat_timeout,
                brute_force_cost=brute_force_cost,
                brute_force_ms=brute_force_ms,
            )
            if maxsat_error is not None:
                row["maxsat_error"] = maxsat_error
            rows.append(row)
            existing_names.add(name)
            out["bounds"] = rows
            save(out)
            print(
                f"[bounds] {name} m={m} n={n} lp={lp_bound} "
                f"greedy={greedy_cost} rollout={greedy_rollout_cost} "
                f"maxsat={maxsat_cost} brute={brute_force_cost}",
                flush=True,
            )


# ------------------------------------------------------- tsp_approx (EggNOG)
# (name, taxid, top_n species, max_ogs) — full-size conversions kept separate
# from the small shared-cache CSVs (<name>_<max_ogs>x<top_n>.csv).
_TSP_EGGNOG = (
    ("eggnog_saccharomyces_4890", 4890, 20, 800),
    ("eggnog_bacillus_1386", 1386, 15, 500),
)


def tsp_eggnog_bench(out: dict) -> None:
    """tsp_approx (Christofides/CBM reduction) vs other UBs on EggNOG large data.

    Separate ``tsp_approx_eggnog`` key so reruns never clash with ``bounds``;
    resumable per dataset.
    """
    from real_data import (
        _CACHE_DIR,
        _convert_eggnog_members_to_csv,
        _fetch_eggnog_members,
        _parse_tsv,
    )

    from mingap import get

    rows: list[dict] = list(out.get("tsp_approx_eggnog", []))
    existing = {row["name"] for row in rows}
    solvers = {}
    for solver_name in ("tsp_approx", "greedy", "lp_round", "lp_relax"):
        try:
            solvers[solver_name] = get(solver_name)
        except ImportError:
            solvers[solver_name] = None

    for name, taxid, top_n, max_ogs in _TSP_EGGNOG:
        if name in existing:
            print(f"[tsp_eggnog] {name}: already in JSON, skipping", flush=True)
            continue
        target = _CACHE_DIR / f"{name}_{max_ogs}x{top_n}.csv"
        if not target.exists() and (
            _fetch_eggnog_members(taxid) is None
            or not _convert_eggnog_members_to_csv(taxid, top_n, max_ogs, target)
        ):
            print(f"[tsp_eggnog] {name}: fetch/convert failed, skipping", flush=True)
            continue
        matrix, _, _ = _parse_tsv(target.read_text(encoding="utf-8"))
        m = len(matrix)
        n = len(matrix[0]) if m else 0

        row: dict[str, object] = dict(name=name, m=m, n=n)
        for solver_name, solver in solvers.items():
            key = "lp_relax_lb" if solver_name == "lp_relax" else f"{solver_name}_ub"
            if solver is None:
                row[key] = None
                row[f"{solver_name}_ms"] = None
                continue
            t0 = time.perf_counter()
            try:
                row[key] = solver.solve(matrix).cost
            except Exception as exc:  # noqa: BLE001
                print(f"[tsp_eggnog] {solver_name} failed: {exc!r}", flush=True)
                row[key] = None
            row[f"{solver_name}_ms"] = (time.perf_counter() - t0) * 1000.0
        rows.append(row)
        existing.add(name)
        out["tsp_approx_eggnog"] = rows
        save(out)
        print(
            f"[tsp_eggnog] {name} m={m} n={n} "
            + " ".join(f"{s}={row.get('lp_relax_lb' if s == 'lp_relax' else f'{s}_ub')}" for s in solvers),
            flush=True,
        )


# ----------------------------------------------------------------- Real data
def _lp_cost_in_parent(matrix, solver_name: str) -> tuple[float | None, float | None]:
    """Compute an LP-derived bound (``lp_relax`` / ``lp_round``) in the parent.

    Also used for ``greedy_rollout``, which spawns its own process pool and
    therefore cannot run inside a daemonic pebble worker.

    Pebble worker timeouts discard *all* worker results, so a worker
    that ran MaxSAT first loses both the MaxSAT cost AND the LP cost
    when it hits the timeout. Computing LP in the parent (where
    the timeout doesn't apply) makes the LP values survive any
    MaxSAT timeout.
    """
    try:
        from mingap import get as _mingap_get
        solver = _mingap_get(solver_name)
    except Exception:
        return None, None
    import time as _time
    t0 = _time.perf_counter()
    try:
        res = solver.solve(matrix)
        cost = float(res.cost)
    except Exception:
        cost = None
    return cost, (_time.perf_counter() - t0) * 1000.0


def _real_solve(payload):
    """Worker: solve one small-tier real matrix with brute + maxsat.

    Brute force runs only for n <= 10. LP relax and LP round are
    computed in the parent (see ``_lp_cost_in_parent``) so their
    bounds survive any worker timeout.
    """
    import time

    from mingap import solve

    name, matrix = payload
    m = len(matrix)
    n = len(matrix[0]) if m else 0

    bf_cost = None
    bf_ms = None
    if n <= 10:
        t0 = time.perf_counter()
        bf_res = solve(matrix, solver="brute_force")
        bf_cost = bf_res.cost
        bf_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    ms_res = solve(matrix, solver="maxsat")
    ms_ms = (time.perf_counter() - t0) * 1000.0

    return dict(
        name=name, m=m, n=n, scale="small",
        brute_force_cost=bf_cost,
        brute_force_ms=bf_ms,
        maxsat_cost=ms_res.cost,
        maxsat_ms=ms_ms,
        maxsat_timeout=False,
    )


def _real_solve_large(payload):
    """Worker: solve one large-tier real matrix with greedy only.

    MaxSAT is skipped on the large tier (it would hit the timeout); the
    row records ``maxsat_cost=None, maxsat_timeout=True``. LP relax and
    LP round are computed in the parent (see ``_lp_cost_in_parent``) so
    their bounds are always present.
    """
    import time

    from mingap import solve

    name, matrix = payload
    m = len(matrix)
    n = len(matrix[0]) if m else 0

    t0 = time.perf_counter()
    g_res = solve(matrix, solver="greedy")
    g_ms = (time.perf_counter() - t0) * 1000.0

    return dict(
        name=name, m=m, n=n, scale="large",
        brute_force_cost=None,
        brute_force_ms=None,
        maxsat_cost=None,
        maxsat_ms=None,
        maxsat_timeout=True,
        greedy_cost=g_res.cost,
        greedy_ms=g_ms,
    )


def _row_is_incomplete(row: dict) -> bool:
    """A real-data row is *incomplete* when every solver cost is missing.

    Such rows are evicted from the resumable set so a re-run can
    repopulate them. Rows with at least one solver cost present
    (e.g., brute_force_cost or lp_relax_cost) are kept as-is.
    """
    cost_keys = (
        "brute_force_cost", "maxsat_cost", "lp_relax_cost", "lp_round_cost",
    )
    return all(row.get(k) is None for k in cost_keys)


def real_data_bench(out: dict) -> None:
    """MinGap on real biological matrices.

    Small tier (n <= 12): brute-force (n <= 10) + MaxSAT. LP relax (in parent).
    Large tier (n > 12): greedy upper bound + LP relax lower bound only;
    MaxSAT is skipped (would timeout) and recorded as
    ``maxsat_cost=None, maxsat_timeout=True``.
    LP relax gives a continuous lower bound and LP round a rounded upper
    bound; both are computed in the
    parent process so they survive any worker timeout. Resumable:
    incomplete rows in the persisted ``real`` array (no solver cost
    present) are dropped and re-solved.
    """
    from concurrent.futures import TimeoutError

    from pebble import ProcessPool

    from real_data import load_all_real

    matrices = load_all_real()
    fresh_rows = [r for r in out.get("real", []) if not _row_is_incomplete(r)]
    rows: list[dict] = list(fresh_rows)
    existing_names = {r.get("name") for r in rows}
    if not matrices:
        print("[real] no real datasets loaded (offline or fetch failed)", flush=True)
        out["real"] = rows
        save(out)
        return

    timeout = 60.0
    with ProcessPool(max_workers=1) as pool:
        for name, matrix, _, _ in matrices:
            if name in existing_names:
                print(f"[real] {name}: already in JSON, skipping", flush=True)
                continue
            m = len(matrix)
            n = len(matrix[0]) if m else 0
            # LP relax + LP round in parent — survive any worker timeout.
            lp_cost, lp_ms = _lp_cost_in_parent(matrix, "lp_relax")
            lr_cost, lr_ms = _lp_cost_in_parent(matrix, "lp_round")
            # Rollout also runs in the parent: it spawns its own process
            # pool, which daemonic pebble workers cannot do.
            gr_cost, gr_ms = _lp_cost_in_parent(matrix, "greedy_rollout")
            payload = (name, matrix)
            worker = _real_solve if n <= 12 else _real_solve_large
            scale = "small" if n <= 12 else "large"
            fut = pool.schedule(worker, args=[payload], timeout=timeout)
            try:
                row = fut.result()
            except TimeoutError:
                row = dict(
                    name=name, m=m, n=n, scale=scale,
                    brute_force_cost=None, brute_force_ms=None,
                    maxsat_cost=None, maxsat_ms=None,
                    maxsat_timeout=True,
                    greedy_cost=None, greedy_ms=None,
                    lp_relax_cost=lp_cost,
                    lp_relax_ms=lp_ms,
                    lp_round_cost=lr_cost,
                    lp_round_ms=lr_ms,
                    greedy_rollout_cost=gr_cost,
                    greedy_rollout_ms=gr_ms,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[real] {name} worker error: {e!r}", flush=True)
                row = dict(
                    name=name, m=m, n=n, scale=scale,
                    brute_force_cost=None, brute_force_ms=None,
                    maxsat_cost=None, maxsat_ms=None,
                    maxsat_timeout=False,
                    greedy_cost=None, greedy_ms=None,
                    lp_relax_cost=lp_cost,
                    lp_relax_ms=lp_ms,
                    lp_round_cost=lr_cost,
                    lp_round_ms=lr_ms,
                    greedy_rollout_cost=gr_cost,
                    greedy_rollout_ms=gr_ms,
                    error=repr(e),
                )
            else:
                row["lp_relax_cost"] = lp_cost
                row["lp_relax_ms"] = lp_ms
                row["lp_round_cost"] = lr_cost
                row["lp_round_ms"] = lr_ms
                row["greedy_rollout_cost"] = gr_cost
                row["greedy_rollout_ms"] = gr_ms
            rows.append(row)
            existing_names.add(name)
            print(
                f"[real] {name} scale={row['scale']} m={row['m']} n={row['n']} "
                f"bf_cost={row['brute_force_cost']} bf_ms={row['brute_force_ms']} "
                f"ms_cost={row['maxsat_cost']} ms_ms={row['maxsat_ms']} "
                f"lp_cost={row['lp_relax_cost']} lp_ms={row['lp_relax_ms']} "
                f"lr_cost={row['lp_round_cost']} lr_ms={row['lp_round_ms']}",
                flush=True,
            )
            out["real"] = rows
            save(out)

# ------------------------------------------- tsp_approx vs exact (small n)
def _tsp_exact_aggregate(entries: list[dict]) -> dict:
    """Aggregate approximation stats over all ``tsp_vs_exact`` entries."""

    def mean(key: str):
        vals = [e[key] for e in entries if e.get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    return dict(
        n_entries=len(entries),
        tsp_mean_ratio=mean("tsp_ratio"),
        greedy_mean_ratio=mean("greedy_ratio"),
        greedy_rollout_mean_ratio=mean("greedy_rollout_ratio"),
        lp_round_mean_ratio=mean("lp_round_ratio"),
        tsp_beats_greedy_count=sum(1 for e in entries if e["tsp_cost"] < e["greedy_cost"]),
        greedy_beats_tsp_count=sum(1 for e in entries if e["greedy_cost"] < e["tsp_cost"]),
        tie_count=sum(1 for e in entries if e["tsp_cost"] == e["greedy_cost"]),
    )


def tsp_vs_exact_bench(out: dict) -> None:
    """tsp_approx / greedy / greedy_rollout / lp_round vs the exact optimum.

    Small matrices only (n <= 10): brute force gives the exact optimum and
    MaxSAT (exact at n <= 12) cross-checks it. Ratios are cost / bf_cost and
    are skipped (None) when bf_cost is 0. Resumable: entries are keyed by
    name and skipped when already present; the aggregate is recomputed from
    all entries after every addition.
    """
    import nb1p

    from mingap import get

    entries: list[dict] = list(out.get("tsp_vs_exact", {}).get("entries", []))
    existing = {e["name"] for e in entries}

    instances: list[tuple[str, list]] = []
    for gen_name, gen, gen_offset in (("random", random_matrix, 0), ("satisfiable", sat_matrix, 1)):
        for t in range(20):
            n = 3 + (t % 8)  # n = 3..10 round-robin
            instances.append((
                f"fuzz_{gen_name}_n{n}_t{t}",
                gen(2 * n, n, seed=seed_for(t, n, gen_offset=gen_offset)),
            ))
    # Shared cross-check fixtures (mirrors tests/test_mingap.py).
    cross_check = [
        [[1, 0, 1], [0, 1, 1], [1, 1, 0]],
        [[1, 0, 0, 1]],
        [[1, 0, 1, 0, 1]],
        [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]],
        nb1p.paper_example(),
    ]
    for i, mat in enumerate(cross_check):
        instances.append((f"cross_check_{i}", mat))

    bf = get("brute_force")
    maxsat = get("maxsat")
    tsp = get("tsp_approx")
    greedy = get("greedy")
    rollout = get("greedy_rollout")
    try:
        lp_round = get("lp_round")
    except ImportError:
        lp_round = None

    for name, mat in instances:
        if name in existing:
            print(f"[tsp_exact] {name}: already in JSON, skipping", flush=True)
            continue
        n = len(mat[0]) if mat else 0
        entry: dict[str, object] = dict(
            name=name, n=n,
            bf_cost=bf.solve(mat).cost,
            maxsat_cost=maxsat.solve(mat).cost if n <= 12 else None,
            tsp_cost=tsp.solve(mat).cost,
            greedy_cost=greedy.solve(mat).cost,
            greedy_rollout_cost=rollout.solve(mat).cost,
            lp_round_cost=lp_round.solve(mat).cost if lp_round is not None else None,
        )
        exact = entry["bf_cost"]
        for key in ("tsp", "greedy", "greedy_rollout", "lp_round"):
            cost = entry[f"{key}_cost"]
            entry[f"{key}_ratio"] = round(cost / exact, 4) if exact and cost is not None else None
        entries.append(entry)
        existing.add(name)
        out["tsp_vs_exact"] = {"entries": entries, "aggregate": _tsp_exact_aggregate(entries)}
        save(out)
        print(
            f"[tsp_exact] {name} n={n} bf={entry['bf_cost']} ms={entry['maxsat_cost']} "
            f"tsp={entry['tsp_cost']} greedy={entry['greedy_cost']} "
            f"rollout={entry['greedy_rollout_cost']} lp={entry['lp_round_cost']}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["sat", "maxsat", "fuzz", "fuzz10", "syntesim", "bounds", "tsp_eggnog", "tsp_exact", "real", "all"])
    args = ap.parse_args()

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    parts = (
        ["sat", "maxsat", "fuzz", "fuzz10", "syntesim", "bounds", "real"] if args.part == "all" else [args.part]
    )
    for part in parts:
        print(f"=== part {part} ===", flush=True)
        bench = {
            "sat": sat_scaling,
            "maxsat": maxsat_scaling,
            "fuzz": fuzz_validate,
            "fuzz10": fuzz_validate_n3_n10,
            "syntesim": syntesim_instances,
            "bounds": bounds_bench,
            "tsp_eggnog": tsp_eggnog_bench,
            "tsp_exact": tsp_vs_exact_bench,
            "real": real_data_bench,
        }
        bench[part](out)
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()