"""
Event callback system for PyControls components.

Provides a mixin class that any estimator, controller, or solver can inherit
to support pre/post-step callbacks without modifying core algorithm logic.

Usage:
    ekf = ExtendedKalmanFilter(...)
    ekf.register_hook("on_update_end", lambda **kw: print(f"x_hat = {kw['state']}"))
"""

from typing import Any, Callable, Dict, FrozenSet, List


class HookMixin:
    """Mixin that adds event hook support to any class."""

    VALID_EVENTS: FrozenSet[str] = frozenset({
        "on_predict_begin",
        "on_predict_end",
        "on_update_begin",
        "on_update_end",
        "on_compute_begin",
        "on_compute_end",
        "on_reset",
        "on_convergence",
        "on_divergence",
    })

    def _init_hooks(self) -> None:
        """Call this in __init__ of any class using HookMixin."""
        self._hooks: Dict[str, List[Callable[..., None]]] = {
            event: [] for event in self.VALID_EVENTS
        }

    def register_hook(self, event: str, callback: Callable[..., None]) -> None:
        """Register a callback for a specific event."""
        if event not in self.VALID_EVENTS:
            raise ValueError(
                f"Unknown event '{event}'. Valid events: {sorted(self.VALID_EVENTS)}"
            )
        if not hasattr(self, "_hooks"):
            self._init_hooks()
        self._hooks[event].append(callback)

    def unregister_hook(self, event: str, callback: Callable[..., None]) -> None:
        """Remove a previously registered callback."""
        if hasattr(self, "_hooks") and event in self._hooks:
            self._hooks[event].remove(callback)

    def _fire(self, event: str, **kwargs: Any) -> None:
        """Fire all callbacks registered for an event. No-op if no hooks registered."""
        if not hasattr(self, "_hooks"):
            return
        for cb in self._hooks.get(event, []):
            cb(**kwargs)
