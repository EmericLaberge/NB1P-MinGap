"""MinGap solver registry — thin facade over the shared generic registry.

Backends register themselves on import; ``get``/``available`` resolve them
lazily. Only ``auto_order`` defaults are resolved automatically; everything
else is explicit (``solver="<name>"``).
"""

from __future__ import annotations

from nb1p.registry import SolverRegistry

registry: SolverRegistry = SolverRegistry(
    backend_modules={
        "brute_force": "mingap._brute_force",
        "maxsat": "mingap._maxsat",
        "greedy": "mingap._greedy",
        "greedy_rollout": "mingap._greedy_rollout",
        "tsp_approx": "mingap._tsp_approx",
    },
    auto_order=("maxsat", "brute_force"),
    extra_map={"maxsat": "sat", "tsp_approx": "tsp"},
    no_backend_msg=(
        "No MinGap solver backend installed. Install one with e.g. "
        "`pip install mingap[sat]`, or use solver='brute_force'."
    ),
)

#: ``@register("maxsat")`` adds a solver class to this problem's registry.
register = registry.register
#: Resolve a backend by name (or ``"auto"`` for the best installed one).
get = registry.get
#: Names of backends whose module can be imported.
available = registry.available
