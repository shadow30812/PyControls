"""
Abstract base classes defining the standard interface for all PyControls components.

These ensure that any estimator, controller, or solver can be used interchangeably
by external integration layers (MATLAB, ROS, Gymnasium, etc.) without knowledge
of the specific algorithm.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BaseEstimator(ABC):
    """
    Standard interface for all state estimators.

    Subclasses: KalmanFilter, ExtendedKalmanFilter, DiscreteExtendedKalmanFilter,
                UnscentedKalmanFilter.

    All estimators follow the predict-update cycle:
        estimator.predict(u, dt)
        x_hat = estimator.update(y_meas)
    """

    @abstractmethod
    def predict(self, u: Any, dt: Optional[float] = None) -> None:
        """Propagate state estimate forward by one time step."""
        ...

    @abstractmethod
    def update(self, y: Union[ArrayLike, NDArray[Any]]) -> NDArray[np.float64]:
        """Incorporate a measurement and return the updated state estimate."""
        ...

    @abstractmethod
    def get_state(self) -> NDArray[np.float64]:
        """Return the current state estimate as a flat 1-D array."""
        ...

    @abstractmethod
    def get_covariance(self) -> NDArray[np.float64]:
        """Return a copy of the current error covariance matrix."""
        ...

    @abstractmethod
    def reset(self, x0: ArrayLike, P0: Optional[ArrayLike] = None) -> None:
        """Reset state and covariance for a new estimation session."""
        ...

    def save_state(self) -> Dict[str, Any]:
        """Serialize runtime state for checkpointing/replay."""
        return {
            "class": type(self).__name__,
            "state": self.get_state().tolist(),
            "covariance": self.get_covariance().tolist(),
        }

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Restore runtime state from a saved checkpoint."""
        self.reset(
            np.array(state_dict["state"]),
            np.array(state_dict["covariance"]),
        )


class BaseController(ABC):
    """
    Standard interface for all controllers.

    Subclasses: PIDController, ModelPredictiveControl.

    All controllers expose a unified compute() method:
        u = controller.compute(state, reference, dt)
    """

    @abstractmethod
    def compute(
        self,
        state: NDArray[np.float64],
        reference: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Compute control output given current state and reference."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state (integral terms, warm-starts, etc.)."""
        ...

    def save_state(self) -> Dict[str, Any]:
        """Serialize controller runtime state."""
        return {"class": type(self).__name__}

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Restore controller state from checkpoint."""
        self.reset()


class BaseSolver(ABC):
    """
    Standard interface for all discrete-time solvers.

    Subclasses: ExactSolver.
    """

    @abstractmethod
    def step(self, u: Any) -> Union[float, NDArray[np.float64]]:
        """Advance the system by one discrete time step."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset solver state to initial conditions."""
        ...

    def save_state(self) -> Dict[str, Any]:
        """Serialize solver runtime state."""
        return {"class": type(self).__name__}

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Restore solver state from checkpoint."""
        self.reset()
