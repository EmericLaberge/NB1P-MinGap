"""MinGap via PySAT RC2 (MaxSAT).
Assignment-matrix encoding with a running has_one_in_block tracker.
Cost per row = (blocks of {1,2} with ≥ 1 real loss) − 1.
Requires: pip install mininv[sat].
"""

from __future__ import annotations

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from mingap.score import block_cost
from nb1p import validate_ternary
from mingap._registry import register
from mingap.types import Matrix, OptimizeResult


@register("maxsat")
class MaxSATSolver:
    """MaxSAT solver minimising the inversion-aware block cost."""

    def solve(self, matrix: Matrix) -> OptimizeResult:
        _, n = validate_ternary(matrix)
        if n <= 1:
            return OptimizeResult(list(range(n)), 0)


        rows_info: list[tuple[list[int], list[int]]] = []
        for row in matrix:
            loss = [j for j in range(n) if row[j] in (1, 2)]
            ones = [j for j in range(n) if row[j] == 1]
            rows_info.append((loss, ones))

        # Skip rows with fewer than 2 ones (they always contribute 0 cost)
        active = [(loss, ones)
                  for loss, ones in rows_info
                  if len(ones) >= 2]

        if not active:
            return OptimizeResult(list(range(n)), 0)

        wcnf = WCNF()
        counter = [0]

        def new_var() -> int:
            counter[0] += 1
            return counter[0]

        # ── Assignment matrix x[j][p]: column j placed at position p ─────
        x = [[new_var() for _ in range(n)] for _ in range(n)]

        # Each position holds exactly one column (exactly-one constraint)
        for p in range(n):
            wcnf.append([x[j][p] for j in range(n)])
            for a in range(n):
                for b in range(a + 1, n):
                    wcnf.append([-x[a][p], -x[b][p]])
        # Each column is placed at exactly one position (exactly-one constraint)
        for j in range(n):
            wcnf.append([x[j][p] for p in range(n)])
            for a in range(n):
                for b in range(a + 1, n):
                    wcnf.append([-x[j][a], -x[j][b]])

        # Symmetry break: col 0 placed before col 1 in the permutation
        for p0 in range(n):
            for p1 in range(p0):
                wcnf.append([-x[0][p0], -x[1][p1]])

        # ── Per-row block counting ──────────────────────────────────────
        for loss_cols, one_cols in active:

            # at_loss[p] ⟷ ∃j∈loss_cols : x[j][p]
            # "position p contains a column where this row has a loss (1 or 2)"
            at_loss = [0] * n
            for p in range(n):
                v = new_var()
                at_loss[p] = v
                wcnf.append([-v] + [x[j][p] for j in loss_cols])
                for j in loss_cols:
                    wcnf.append([-x[j][p], v])

            # at_one[p] ⟷ ∃j∈one_cols : x[j][p]
            # "position p contains a column where this row has a real loss (1)"
            at_one = [0] * n
            for p in range(n):
                v = new_var()
                at_one[p] = v
                wcnf.append([-v] + [x[j][p] for j in one_cols])
                for j in one_cols:
                    wcnf.append([-x[j][p], v])

            # has_one[p]: "the current active block ending at p has seen ≥ 1 real 1"
            # has_one[p] ⟷ at_loss[p] ∧ (at_one[p] ∨ has_one[p-1])
            has_one = [0] * n

            # p = 0: only at_loss[0] ∧ at_one[0] can start a block
            has_one_first = new_var()
            has_one[0] = has_one_first
            wcnf.append([-has_one_first, at_loss[0]])          # has_one[0] → at_loss[0]
            wcnf.append([-has_one_first, at_one[0]])            # has_one[0] → at_one[0]
            wcnf.append([-at_loss[0], -at_one[0], has_one_first])  # at_loss ∧ at_one → has_one
            wcnf.append([at_loss[0], -has_one_first])           # ¬at_loss → ¬has_one

            for p in range(1, n):
                has_one_p = new_var()
                has_one[p] = has_one_p

                # has_one[p] ⟷ at_loss[p] ∧ (at_one[p] ∨ has_one[p-1])
                wcnf.append([-has_one_p, at_loss[p]])                   # has_one → at_loss
                wcnf.append([-has_one_p, at_one[p], has_one[p - 1]])    # has_one → at_one ∨ prev has_one
                wcnf.append([-at_loss[p], -at_one[p], has_one_p])       # at_loss ∧ at_one → has_one
                wcnf.append([-at_loss[p], -has_one[p - 1], has_one_p])  # at_loss ∧ prev has_one → has_one

            # end_with_one[p]: a counted block ends at position p
            # end_with_one[p] ⟷ has_one[p] ∧ (p = n-1 ∨ ¬has_one[p+1])
            end_with_one = [0] * n
            for p in range(n - 1):
                v = new_var()
                end_with_one[p] = v
                # v ⟷ has_one[p] ∧ ¬has_one[p+1]
                wcnf.append([-v, has_one[p]])              # v → has_one[p]
                wcnf.append([-v, -has_one[p + 1]])         # v → ¬has_one[p+1]
                wcnf.append([-has_one[p], has_one[p + 1], v])  # has_one[p] ∧ ¬has_one[p+1] → v
            end_with_one[n - 1] = has_one[n - 1]

            # seen_end[p]: some end_with_one occurred at or before position p
            # Soft clauses penalise extra block ends beyond the first → minimises inversions
            seen_end = [0] * n
            seen_end[0] = end_with_one[0]

            for p in range(1, n):
                seen_end_p = new_var()
                seen_end[p] = seen_end_p
                # seen_end[p] ⟷ seen_end[p-1] ∨ end_with_one[p]
                wcnf.append([-seen_end_p, seen_end[p - 1], end_with_one[p]])
                wcnf.append([-seen_end[p - 1], seen_end_p])
                wcnf.append([-end_with_one[p], seen_end_p])

            for p in range(1, n):
                extra = new_var()
                # extra[p] ⟷ end_with_one[p] ∧ seen_end[p-1]
                # "a counted block ends here AND we've already counted one before"
                wcnf.append([-extra, end_with_one[p]])
                wcnf.append([-extra, seen_end[p - 1]])
                wcnf.append([-end_with_one[p], -seen_end[p - 1], extra])

                # Soft: prefer no extra block end (each adds 1 to cost)
                wcnf.append([-extra], weight=1)

        # ── Solve ───────────────────────────────────────────────────────
        with RC2(wcnf) as rc2:
            model = rc2.compute()
            if model is None:
                identity = list(range(n))
                return OptimizeResult(identity, block_cost(matrix, identity))
            truth = {abs(v): v > 0 for v in model}
            perm = [0] * n
            for p in range(n):
                for j in range(n):
                    if truth.get(x[j][p], False):
                        perm[p] = j
                        break
            return OptimizeResult(perm, block_cost(matrix, perm))
