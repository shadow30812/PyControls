"""
Handle-based object registry for external integrations.

Provides lifecycle management (register, get, reset, destroy) so that
external systems (MATLAB, ROS, Gymnasium) hold lightweight string handles
instead of raw Python object references.

Multi-agent ready: use namespaced handles like "drone_1/ekf", "drone_2/mpc".
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from core.base import BaseController, BaseEstimator, BaseSolver

ManagedObject = Union[BaseEstimator, BaseController, BaseSolver]


class ControllerManager:
    """Registry for estimators, controllers, and solvers."""

    def __init__(self) -> None:
        self._registry: Dict[str, ManagedObject] = {}
        self._counter: int = 0

    def register(self, obj: ManagedObject, name: Optional[str] = None) -> str:
        """
        Register an object and return its handle.

        Args:
            obj: Any BaseEstimator, BaseController, or BaseSolver instance.
            name: Optional custom handle. Auto-generated if omitted.

        Returns:
            The string handle used to reference this object.
        """
        if name is None:
            name = f"{type(obj).__name__}_{self._counter}"
            self._counter += 1
        if name in self._registry:
            raise KeyError(f"Handle '{name}' already registered. Use destroy() first.")
        self._registry[name] = obj
        return name

    def get(self, handle: str) -> ManagedObject:
        """Retrieve a registered object by handle."""
        if handle not in self._registry:
            raise KeyError(
                f"Handle '{handle}' not found. Registered: {list(self._registry.keys())}"
            )
        return self._registry[handle]

    def reset(self, handle: str) -> None:
        """Reset a specific registered object."""
        obj = self.get(handle)
        if isinstance(obj, BaseEstimator):
            # Estimator reset requires x0; re-reset to current state shape
            state = obj.get_state()
            obj.reset(state * 0.0)  # Zero state, keep dimensions
        else:
            obj.reset()

    def reset_all(self) -> None:
        """Reset all registered objects."""
        for handle in list(self._registry.keys()):
            self.reset(handle)

    def destroy(self, handle: str) -> None:
        """Remove an object from the registry."""
        if handle not in self._registry:
            raise KeyError(f"Handle '{handle}' not found.")
        del self._registry[handle]

    def destroy_all(self) -> None:
        """Remove all objects from the registry."""
        self._registry.clear()
        self._counter = 0

    def list_handles(self) -> Dict[str, str]:
        """List all registered handles with their class names."""
        return {k: type(v).__name__ for k, v in self._registry.items()}

    def save_all(self) -> Dict[str, Dict[str, Any]]:
        """Save state of all registered objects that support serialization."""
        states: Dict[str, Dict[str, Any]] = {}
        for handle, obj in self._registry.items():
            if hasattr(obj, "save_state"):
                states[handle] = obj.save_state()
        return states

    def load_all(self, states: Dict[str, Dict[str, Any]]) -> None:
        """Load state into all registered objects from a saved checkpoint."""
        for handle, state_dict in states.items():
            if handle in self._registry and hasattr(
                self._registry[handle], "load_state"
            ):
                self._registry[handle].load_state(state_dict)

    def step_estimator(
        self, handle: str, u: Any, y: Any, dt: float
    ) -> NDArray[np.float64]:
        """Convenience: predict + update in one call. Returns updated state."""
        est = self.get(handle)
        if not isinstance(est, BaseEstimator):
            raise TypeError(f"Handle '{handle}' is not an estimator.")
        est.predict(u, dt)
        return est.update(y)

    def step_controller(
        self, handle: str, state: Any, reference: Any, dt: float
    ) -> NDArray[np.float64]:
        """Convenience: compute control output in one call."""
        ctrl = self.get(handle)
        if not isinstance(ctrl, BaseController):
            raise TypeError(f"Handle '{handle}' is not a controller.")
        return ctrl.compute(state, reference, dt)
