"""
Configuration dataclasses for PyControls components.

Usage:
    cfg = EKFConfig(Q=..., R=..., x0=...)

    # Or load from JSON:
    import json
    with open("ekf_config.json") as f:
        cfg = EKFConfig(**json.load(f))
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class EKFConfig:
    """Configuration for ExtendedKalmanFilter."""
    Q: ArrayLike = field(default_factory=lambda: np.eye(1))
    R: ArrayLike = field(default_factory=lambda: np.eye(1))
    x0: ArrayLike = field(default_factory=lambda: np.zeros(1))
    p_init_scale: float = 0.1


@dataclass
class UKFConfig:
    """Configuration for UnscentedKalmanFilter."""
    Q: ArrayLike = field(default_factory=lambda: np.eye(1))
    R: ArrayLike = field(default_factory=lambda: np.eye(1))
    x0: ArrayLike = field(default_factory=lambda: np.zeros(1))
    P0: ArrayLike = field(default_factory=lambda: np.eye(1))
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0


@dataclass
class PIDConfig:
    """Configuration for PIDController."""
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    derivative_on_measurement: bool = True
    output_limits: Tuple[Optional[float], Optional[float]] = (None, None)
    integral_limits: Tuple[Optional[float], Optional[float]] = (None, None)
    tau: float = 0.02


@dataclass
class MPCConfig:
    """Configuration for ModelPredictiveControl."""
    horizon: int = 10
    dt: float = 0.1
    Q: Optional[ArrayLike] = None
    R: Optional[ArrayLike] = None
    u_min: float = -10.0
    u_max: float = 10.0


@dataclass
class SolverConfig:
    """Configuration for ExactSolver."""
    dt: float = 0.01
