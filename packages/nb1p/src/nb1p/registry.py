"""Generic solver registry — lazy backend loading shared by ``nb1p`` and ``mingap``.

Each problem instantiates :class:`SolverRegistry` with its backend-module map,
``auto`` preference order, and pip-extra map, then re-exports ``register``,
``get`` and ``available``. Backends register themselves with
``@registry.register(name)`` on import; the backend module is imported lazily,
only when that backend is actually requested. This keeps the core importable
with no third-party dependencies.
"""

from __future__ import annotations

import importlib
from typing import Callable


class SolverRegistry:
    """Backend table and resolver for one problem family.

    Args:
        backend_modules: backend name -> dotted module path, imported on demand.
        auto_order: backend names tried, best first, when ``name="auto"``.
        extra_map: backend name -> pip extra, used in the install hint when the
            backend is known but its optional dependency is missing. Names not
            present here fall back to ``"all"``.
        no_backend_msg: message raised when ``auto`` resolves nothing.
    """
    _solvers: dict[str, type]
    _backend_modules: dict[str, str]
    _auto_order: tuple[str, ...]
    _extra_map: dict[str, str]
    _no_backend_msg: str


    def __init__(
        self,
        *,
        backend_modules: dict[str, str],
        auto_order: tuple[str, ...],
        extra_map: dict[str, str],
        no_backend_msg: str,
    ) -> None:
        self._solvers: dict[str, type] = {}
        self._backend_modules = backend_modules
        self._auto_order = auto_order
        self._extra_map = extra_map
        self._no_backend_msg = no_backend_msg

    def register(self, name: str) -> Callable[[type], type]:
        """Decorator: ``@registry.register("sat")`` adds the solver class."""
        def decorator(cls: type) -> type:
            self._solvers[name] = cls
            return cls

        return decorator

    def _try_load(self, name: str) -> bool:
        """Import the backend module for *name* if it is not loaded yet."""
        if name in self._solvers:
            return True
        module = self._backend_modules.get(name)
        if module is None:
            return False
        try:
            _ = importlib.import_module(module)
        except ImportError:
            return False
        return name in self._solvers

    def _extra_for(self, name: str) -> str:
        return self._extra_map.get(name, "all")

    def get(self, name: str = "auto"):
        """Instantiate a solver backend.

        Args:
            name: backend name, or ``"auto"`` to pick the best available.

        Raises:
            RuntimeError: if ``"auto"`` and no backend is installed.
            ValueError:   if *name* is unknown.
            ImportError:  if a known backend's optional dependency is missing.
        """
        if name == "auto":
            for pref in self._auto_order:
                if self._try_load(pref):
                    return self._solvers[pref]()
            raise RuntimeError(self._no_backend_msg)

        if not self._try_load(name):
            if name in self._backend_modules:
                raise ImportError(
                    f"Backend '{name}' is known but its optional dependency is not installed; try `pip install mininv[{self._extra_for(name)}]`."
                )
            available_names = sorted(set(self._solvers) | set(self._backend_modules))
            raise ValueError(
                f"Unknown solver '{name}'. Available: {available_names}"
            )
        return self._solvers[name]()

    def available(self) -> list[str]:
        """Names of backends whose module can actually be imported.

        A backend may be *loadable* (its module imports and registers a class)
        without being *functional* — e.g. a stub that raises
        ``NotImplementedError``. Such stubs are still reported here.
        """
        return [name for name in self._backend_modules if self._try_load(name)] + [
            name for name in self._solvers if name not in self._backend_modules
        ]
