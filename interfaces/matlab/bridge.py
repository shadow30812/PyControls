"""
Thin MATLAB-to-PyControls bridge.

This module provides the entry point for MATLAB's Python Engine API.
It converts MATLAB arrays to NumPy and forwards all calls to the
ControllerManager. Contains ZERO control logic.

MATLAB usage:
    py.importlib.import_module('interfaces.matlab.bridge');
    h = py.interfaces.matlab.bridge.create_ekf(f, h, Q, R, x0);
    py.interfaces.matlab.bridge.predict(h, u, dt);
    x_hat = double(py.interfaces.matlab.bridge.update(h, y));
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.control_utils import PIDController, dlqr, solve_discrete_riccati
from core.ekf import ExtendedKalmanFilter
from core.ekf_discrete import DiscreteExtendedKalmanFilter
from core.estimator import KalmanFilter
from core.math_utils import jacobian
from core.mpc import ModelPredictiveControl
from core.solver import ExactSolver, discretize_zoh
from core.ukf import UnscentedKalmanFilter
from interfaces.manager import ControllerManager

try:
    from modules.physics_engine import rk4_fixed_step
except ImportError:
    rk4_fixed_step = None  # type: ignore[assignment]

# Module-level manager instance for MATLAB sessions.
# MATLAB holds string handles, this module owns the objects.
_mgr = ControllerManager()


# ── Factory functions ──────────────────────────────────────────────────────


def create_kf(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    Q: ArrayLike,
    R: ArrayLike,
    x0: ArrayLike,
    name: Optional[str] = None,
) -> str:
    """Create a Kalman Filter and return its handle."""
    obj = KalmanFilter(
        np.asarray(A),
        np.asarray(B),
        np.asarray(C),
        np.asarray(Q),
        np.asarray(R),
        x0,
    )
    return _mgr.register(obj, name)


def create_ekf(
    f_dynamics: Callable,
    h_measurement: Callable,
    Q: ArrayLike,
    R: ArrayLike,
    x0: ArrayLike,
    p_init_scale: float = 0.1,
    name: Optional[str] = None,
) -> str:
    """Create an Extended Kalman Filter and return its handle."""
    obj = ExtendedKalmanFilter(
        f_dynamics,
        h_measurement,
        np.asarray(Q),
        np.asarray(R),
        np.asarray(x0),
        p_init_scale,
    )
    return _mgr.register(obj, name)


def create_discrete_ekf(
    f: Callable,
    h: Callable,
    Q: ArrayLike,
    R: ArrayLike,
    x0: ArrayLike,
    dt: float,
    name: Optional[str] = None,
) -> str:
    """Create a Discrete Extended Kalman Filter and return its handle."""
    obj = DiscreteExtendedKalmanFilter(f, h, Q, R, x0, dt)
    return _mgr.register(obj, name)


def create_ukf(
    f_dynamics: Callable,
    h_measurement: Callable,
    Q: ArrayLike,
    R: ArrayLike,
    x0: ArrayLike,
    P0: ArrayLike,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
    name: Optional[str] = None,
) -> str:
    """Create an Unscented Kalman Filter and return its handle."""
    obj = UnscentedKalmanFilter(
        f_dynamics,
        h_measurement,
        Q,
        R,
        x0,
        P0,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )
    return _mgr.register(obj, name)


def create_pid(
    Kp: float,
    Ki: float,
    Kd: float,
    output_limits: Tuple[Optional[float], Optional[float]] = (None, None),
    name: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Create a PID Controller and return its handle."""
    obj = PIDController(Kp, Ki, Kd, output_limits=output_limits, **kwargs)
    return _mgr.register(obj, name)


def create_mpc(
    model_func: Optional[Callable] = None,
    x0: Optional[ArrayLike] = None,
    horizon: int = 10,
    dt: float = 0.1,
    Q: Optional[ArrayLike] = None,
    R: Optional[ArrayLike] = None,
    u_min: float = -10.0,
    u_max: float = 10.0,
    A: Optional[ArrayLike] = None,
    B: Optional[ArrayLike] = None,
    name: Optional[str] = None,
) -> str:
    """Create an MPC controller and return its handle."""
    obj = ModelPredictiveControl(
        model_func=model_func,
        x0=np.asarray(x0) if x0 is not None else None,
        horizon=horizon,
        dt=dt,
        Q=Q,
        R=R,
        u_min=u_min,
        u_max=u_max,
        A=np.asarray(A) if A is not None else None,
        B=np.asarray(B) if B is not None else None,
    )
    return _mgr.register(obj, name)


def create_solver(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    dt: float,
    name: Optional[str] = None,
) -> str:
    """Create a ZOH LTI solver and return its handle."""
    obj = ExactSolver(A, B, C, D, dt)
    return _mgr.register(obj, name)


# ── Lifecycle operations ───────────────────────────────────────────────────


def get(handle: str) -> Any:
    """Retrieve the raw Python object (for direct attribute access from MATLAB)."""
    return _mgr.get(handle)


def reset(handle: str) -> None:
    """Reset a registered object."""
    _mgr.reset(handle)


def destroy(handle: str) -> None:
    """Remove a registered object."""
    _mgr.destroy(handle)


def destroy_all() -> None:
    """Remove all registered objects."""
    _mgr.destroy_all()


def list_handles() -> Dict[str, str]:
    """List all registered handles and their types."""
    return _mgr.list_handles()


def save_all() -> Dict[str, Any]:
    """Save state of all registered objects."""
    return _mgr.save_all()


def load_all(states: Dict[str, Any]) -> None:
    """Load state into all registered objects."""
    _mgr.load_all(states)


# ── Step operations (type-converting wrappers) ─────────────────────────────


def predict(handle: str, u: Any, dt: float) -> None:
    """Estimator predict step (MATLAB array -> NumPy conversion)."""
    _mgr.get(handle).predict(np.asarray(u), float(dt))


def update(handle: str, y: Any) -> NDArray[np.float64]:
    """Estimator update step. Returns state estimate as NumPy array."""
    est = _mgr.get(handle)
    est.update(np.asarray(y))
    return est.get_state()


def estimate(
    handle: str, u: Any, y: Any, dt: float
) -> NDArray[np.float64]:
    """Combined predict + update. Returns state estimate."""
    return _mgr.step_estimator(
        handle, np.asarray(u), np.asarray(y), float(dt)
    )


def control(
    handle: str, state: Any, reference: Any, dt: float
) -> NDArray[np.float64]:
    """Controller compute step. Returns control output."""
    return _mgr.step_controller(
        handle,
        np.asarray(state),
        np.asarray(reference),
        float(dt),
    )


def solver_step(handle: str, u: Any) -> Any:
    """Solver step. Returns output."""
    return _mgr.get(handle).step(np.asarray(u))


# ── Stateless utilities (no handle needed) ──────────────────────────────────


def compute_lqr(
    A: ArrayLike, B: ArrayLike, Q: ArrayLike, R: ArrayLike
) -> NDArray[np.float64]:
    """Compute discrete LQR gain matrix."""
    return dlqr(np.asarray(A), np.asarray(B), np.asarray(Q), np.asarray(R))


def compute_dare(
    A: ArrayLike, B: ArrayLike, Q: ArrayLike, R: ArrayLike
) -> NDArray[np.float64]:
    """Solve discrete algebraic Riccati equation."""
    return solve_discrete_riccati(A, B, Q, R)


def compute_jacobian(func: Callable, x: ArrayLike, *args: Any) -> NDArray[np.float64]:
    """Compute Jacobian via complex-step differentiation."""
    return jacobian(func, np.asarray(x), *args)


def compute_discretize_zoh(
    A: ArrayLike, B: ArrayLike, dt: float
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Discretize continuous (A, B) via Zero-Order Hold."""
    return discretize_zoh(A, B, dt)


def integrate_rk4(
    dynamics_func: Callable,
    x: ArrayLike,
    u: Any,
    dt: float,
    params: dict,
    disturbance: float = 0.0,
) -> NDArray[np.float64]:
    """Single RK4 integration step."""
    if rk4_fixed_step is None:
        raise ImportError("modules.physics_engine not available.")
    return rk4_fixed_step(
        dynamics_func, np.asarray(x), u, dt, params, disturbance
    )
