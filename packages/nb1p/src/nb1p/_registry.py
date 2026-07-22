"""NB1P solver registry — thin facade over the shared generic registry.

Backends (``sat``, ``brute_force``) register themselves on import;
``get``/``available`` resolve them lazily.  ``auto`` picks the best
installed backend in ``(sat, brute_force)`` order.
"""

from __future__ import annotations

from nb1p.registry import SolverRegistry

registry: SolverRegistry = SolverRegistry(
    backend_modules={
        "brute_force": "nb1p._brute_force",
        "sat": "nb1p._sat",
    },
    auto_order=("sat", "brute_force"),
    extra_map={"sat": "sat"},
    no_backend_msg=(
        "No NB1P solver backend installed. Install one with e.g. "
        "`pip install nb1p[sat]`, or use solver='brute_force'."
    ),
)

#: ``@register("sat")`` adds a solver class to this problem's registry.
register = registry.register
#: Resolve a backend by name (or ``"auto"`` for the best installed one).
get = registry.get
#: Names of backends whose module can be imported.
available = registry.available
