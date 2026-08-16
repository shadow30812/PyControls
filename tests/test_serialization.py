import pytest
import numpy as np
from core.estimator import KalmanFilter
from core.ekf import ExtendedKalmanFilter
from core.ukf import UnscentedKalmanFilter
from core.control_utils import PIDController
from core.mpc import ModelPredictiveControl
from core.solver import ExactSolver

def test_kf_serialization():
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [0.1]])
    C = np.array([[1, 0]])
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    x0 = np.array([0.0, 0.0])
    
    kf = KalmanFilter(A, B, C, Q, R, x0)
    for _ in range(10):
        kf.predict(np.array([1.0]))
        kf.update(np.array([0.5]))
    
    state_before = kf.get_state().copy()
    saved = kf.save_state()
    kf.reset(np.zeros(2))
    kf.load_state(saved)
    np.testing.assert_array_almost_equal(kf.get_state(), state_before)

def test_ekf_serialization():
    f = lambda x, u: x
    h = lambda x: x[:1]
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    x0 = np.array([0.0, 0.0])
    
    ekf = ExtendedKalmanFilter(f, h, Q, R, x0)
    for _ in range(10):
        ekf.predict(np.array([1.0]), 0.1)
        ekf.update(np.array([0.5]))
    
    state_before = ekf.get_state().copy()
    saved = ekf.save_state()
    ekf.reset(np.zeros(2))
    ekf.load_state(saved)
    np.testing.assert_array_almost_equal(ekf.get_state(), state_before)

def test_ukf_serialization():
    f = lambda x, u, dt: x
    h = lambda x: x[:1]
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    x0 = np.array([0.0, 0.0])
    P0 = np.eye(2) * 0.1
    
    ukf = UnscentedKalmanFilter(f, h, Q, R, x0, P0, alpha=1e-3, beta=2, kappa=0)
    for _ in range(10):
        ukf.predict(np.array([1.0]), 0.1)
        ukf.update(np.array([0.5]))
    
    state_before = ukf.get_state().copy()
    saved = ukf.save_state()
    ukf.reset(np.zeros(2), P0)
    ukf.load_state(saved)
    np.testing.assert_array_almost_equal(ukf.get_state(), state_before)

def test_pid_serialization():
    pid = PIDController(1.0, 0.1, 0.01)
    for _ in range(5):
        pid.update(1.0, 2.0, 0.01)
    
    saved = pid.save_state()
    assert 'integral_error' in saved
    pid.reset()
    pid.load_state(saved)
    assert pid.integral_error == saved['integral_error']

def test_mpc_serialization():
    f = lambda x, u, dt: x + u * dt
    mpc = ModelPredictiveControl(model_func=f, x0=np.zeros(1), horizon=3, dt=0.1, Q=np.eye(1), R=np.eye(1), u_min=-1, u_max=1)
    
    for _ in range(3):
        mpc.optimize(np.array([0.0]), np.array([1.0]))
    
    saved = mpc.save_state()
    mpc.reset()
    mpc.load_state(saved)
    np.testing.assert_array_almost_equal(mpc.u_seq, saved['u_seq'])

def test_exactsolver_serialization():
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = np.array([[0]])
    dt = 0.1
    
    solver = ExactSolver(A, B, C, D, dt)
    for _ in range(5):
        solver.step(np.array([1.0]))
    
    x_before = solver.x.copy()
    saved = solver.save_state()
    solver.reset()
    solver.load_state(saved)
    np.testing.assert_array_almost_equal(solver.x.flatten(), x_before.flatten())
