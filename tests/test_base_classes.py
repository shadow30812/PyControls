import pytest
import numpy as np
from core.base import BaseEstimator, BaseController, BaseSolver
from core.estimator import KalmanFilter
from core.ekf import ExtendedKalmanFilter
from core.ekf_discrete import DiscreteExtendedKalmanFilter
from core.ukf import UnscentedKalmanFilter
from core.control_utils import PIDController
from core.mpc import ModelPredictiveControl
from core.solver import ExactSolver

def test_kalman_filter_base():
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [0.1]])
    C = np.array([[1, 0]])
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    x0 = np.array([0, 0])
    
    kf = KalmanFilter(A, B, C, Q, R, x0)
    assert isinstance(kf, BaseEstimator)

def test_extended_kalman_filter_base():
    f = lambda x, u: x
    h = lambda x: x[:1]
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    x0 = np.array([0, 0])
    
    ekf = ExtendedKalmanFilter(f, h, Q, R, x0)
    assert isinstance(ekf, BaseEstimator)

def test_discrete_extended_kalman_filter_base():
    f = lambda x, u: x
    h = lambda x: x[:1]
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    x0 = np.array([0, 0])
    dt = 0.1
    
    dekf = DiscreteExtendedKalmanFilter(f, h, Q, R, x0, dt)
    assert isinstance(dekf, BaseEstimator)

def test_unscented_kalman_filter_base():
    f = lambda x, u, dt: x
    h = lambda x: x[:1]
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    x0 = np.array([0, 0])
    P0 = np.eye(2) * 0.1
    
    ukf = UnscentedKalmanFilter(f, h, Q, R, x0, P0, alpha=1e-3, beta=2, kappa=0)
    assert isinstance(ukf, BaseEstimator)

def test_pid_controller_base():
    pid = PIDController(1.0, 0.1, 0.01)
    assert isinstance(pid, BaseController)

def test_mpc_base():
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [0.1]])
    mpc = ModelPredictiveControl(model_func=None, x0=np.zeros(2), horizon=5, dt=0.1, Q=np.eye(2), R=np.eye(1), u_min=-1, u_max=1, A=A, B=B)
    assert isinstance(mpc, BaseController)

def test_exact_solver_base():
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = np.array([[0]])
    dt = 0.1
    
    solver = ExactSolver(A, B, C, D, dt)
    assert isinstance(solver, BaseSolver)
