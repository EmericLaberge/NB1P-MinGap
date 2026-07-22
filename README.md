# NB1P-MinGap

Solvers for two related column-permutation problems on a ternary matrix
`M ∈ {0,1,2}^{m×n}` (`0` = present, `1` = lost, `2` = ancestral loss / wildcard):

- **NB1P** (_Non-Breaking Ones Property_, decision): is there a column permutation
  that makes every row _segmental_ — no `0` trapped between two `1`s?
- **MinGap** (optimisation): minimise the number of **gaps**. A gap is a maximal
  run of consecutive `0`s trapped between two `1`s, and **each gap costs 1
  regardless of its length** (`1 0 0 1` is a single gap). A cost of `0` is exactly
  the case where NB1P is satisfiable.

These arise in parsimony analysis of gene gain/loss on phylogenetic trees.

## The typical workflow

Feed a ternary matrix to `mingap.analyze`: it checks NB1P and, when NB1P fails,
returns a permutation minimising the number of gaps.

```python
import mingap

report = mingap.analyze(matrix)
report.is_nb1p       # True  → matrix is NB1P (report.n_gaps == 0)
report.permutation   # a gap-free ordering if NB1P, else a minimum-gap one
report.n_gaps        # number of gaps under that permutation (0 iff NB1P)
```

A single MinGap solve covers both questions: the optimum has cost `0`
exactly when the matrix is NB1P.

## Install

The repo is a `uv` workspace of four packages — installable individually or
together.

```bash
# From the workspace root, install everything (or a subset):
uv sync                       # core: nb1p, mingap, sglpp, syntesim_bridge
uv sync --extra sat           # + SAT / MaxSAT backend (PySAT)
uv sync --extra sa            # + JIT brute force (numba)
uv sync --extra data          # + DuckDB access
uv sync --extra viz           # + matplotlib rendering
uv sync --extra all           # all of the above
```

The extras are declared in each package's own `pyproject.toml` (e.g.
`mingap[sat]`, `sglpp[viz]`). Solver backends are imported lazily — code that
only calls `nb1p.check` / `mingap.score.block_cost` pulls in no third-party
dependency.

`syntesim_bridge` additionally depends on
[`syntesim`](https://github.com/EmericLaberge/syntesim-mirror) (external
simulator). See [Syntesim bridge](#syntesim-bridge) below for setup.

## Usage

```python
import nb1p
import mingap

M = nb1p.paper_example()

# ── NB1P (decision) ───────────────────────────────────────
ok        = nb1p.check(M)                          # bool, dependency-free
ok, perm  = nb1p.solve(M, solver="sat")           # one witness
perms     = nb1p.solve_all(M)                      # all witnesses

# ── MinGap (optimisation) ─────────────────────────────────
perm, cost = mingap.solve(M, solver="maxsat")
solutions  = mingap.solve_all(M)                  # all optimal permutations

# ── Shared utilities ──────────────────────────────────────
nb1p.verify(M, perm)                  # O(mn) certificate check
mingap.score.block_cost(M, perm)      # block cost (gap count)
```

`solver="auto"` (the default) picks the best installed backend.

### Available backends

| Problem  | Backends (`auto` order: best → fallback) | Extra        |
| -------- | ---------------------------------------- | ------------ |
| `nb1p`   | `sat` → `brute_force`                    | `sat`        |
| `mingap` | `maxsat` → `brute_force`                  | `sat`        |

`brute_force` is always available (pure Python). Inspect what is installed with
`nb1p.available()` / `mingap.available()`.

### Custom backends

The registry is open — register your own solver and select it by name:

```python
from nb1p import register, SolveResult

@register("my_z3")
class Z3Solver:
    def solve(self, matrix) -> SolveResult: ...
    def solve_all(self, matrix) -> list[list[int]]: ...

nb1p.solve(M, solver="my_z3")
```

## CLI

Each problem package exposes its own command-line entry point:

```bash
nb1p '[[1,2,0],[0,1,2]]' --solver sat          # NB1P decision
mingap-cli solve '[[1,2,0],[0,1,2]]' --solver maxsat  # MinGap optimisation
mingap-cli analyze '[[0,1,2],[2,0,1]]'         # NB1P check, then MinGap if needed
```

See `nb1p --help` and `mingap-cli --help` for full options.

## Phylogenetics

`sglpp` builds ternary matrices from Dollo-consistent gene-loss scenarios
on rooted binary trees and searches small trees for NB1P counter-examples.

```python
import sglpp
sglpp.search(max_leaves=4, max_genes=6, visualize=False)
```

## Syntesim bridge

The optional `syntesim_bridge` package feeds a syntesim simulation log
(JSON-lines, as written by the upstream `simulate` script) into the
NB1P / MinGap solvers. The bridge tracks **gains + losses only**;
transfers, duplications, cut/join, and extinctions are dropped.

```python
from syntesim_bridge import load

result = load("simulate.log")
# result.matrix     — list[list[int]] in {0, 1, 2}
# result.leaves     — row labels (species ids)
# result.genes      — column labels (gene ids)
# result.is_nb1p()  — bool, via nb1p.check
```

A CLI wrapper is included for quick inspection:

```bash
python scripts/syntesim_analyze.py simulate.log
python scripts/syntesim_analyze.py simulate.log --solve
```

The `syntesim` package is **not** on PyPI — it lives outside this repo and
must be cloned and made visible to `uv` before `syntesim_bridge` will
install. The current `pyproject.toml` points `uv` at a sibling path:

```toml
# pyproject.toml → [tool.uv.sources]
syntesim = { path = "../../redaction/syntesim-mirror", editable = true }
```

**One-time setup** (clone the mirror alongside this repo):

```bash
# Either clone the mirror at the expected path:
git clone https://github.com/EmericLaberge/syntesim-mirror \
    ../redaction/syntesim-mirror

# Or, if your syntesim lives elsewhere, edit pyproject.toml's [tool.uv.sources]:
#   syntesim = { path = "/your/path/to/syntesim", editable = true }
#   # or, for a published release:
#   syntesim = "1.2.3"

# Then re-sync:
uv sync
```

**Verify**: `python -c "import syntesim; print(syntesim.__file__)"` should
print a path under your chosen directory (not `ImportError`). If
`syntesim_bridge` fails to import after install, this is the cause.

Without `syntesim` available, the rest of the workspace (NB1P, MinGap, SGLPP,
the CLI, all bench scripts) still installs and works — only
`syntesim_bridge.load` requires it.

## Development

```bash
uv sync
pytest
python scripts/verify_reduction.py   # checks NB1P(M)=YES ⟺ min_cost(M)=0
```

## License

MIT.