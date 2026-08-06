import pytest
import numpy as np
from core.estimator import KalmanFilter
from core.ekf import ExtendedKalmanFilter
from core.ekf_discrete import DiscreteExtendedKalmanFilter
from core.ukf import UnscentedKalmanFilter
from core.control_utils import PIDController
from core.mpc import ModelPredictiveControl

def test_estimator_reset():
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [0.1]])
    C = np.array([[1, 0]])
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    x0 = np.array([0.0, 0.0])
    
    kf = KalmanFilter(A, B, C, Q, R, x0)
    for _ in range(5):
        kf.predict(np.array([1.0]))
        kf.update(np.array([0.5]))
    
    new_x0 = np.array([1.0, -1.0])
    kf.reset(new_x0)
    np.testing.assert_array_almost_equal(kf.get_state(), new_x0)

def test_estimator_get_state_covariance():
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [0.1]])
    C = np.array([[1, 0]])
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    x0 = np.array([0.0, 0.0])
    
    kf = KalmanFilter(A, B, C, Q, R, x0)
    state = kf.get_state()
    assert state.ndim == 1
    
    cov = kf.get_covariance()
    cov[0, 0] = 999.0
    cov2 = kf.get_covariance()
    assert cov2[0, 0] != 999.0

def test_pid_reset():
    pid = PIDController(1.0, 0.1, 0.01)
    for _ in range(5):
        pid.update(1.0, 2.0, 0.01)
    
    pid.reset()
    assert pid.integral_error == 0.0

def test_mpc_reset():
    f = lambda x, u, dt: x + u * dt
    mpc = ModelPredictiveControl(model_func=f, x0=np.zeros(1), horizon=3, dt=0.1, Q=np.eye(1), R=np.eye(1), u_min=-1, u_max=1)
    
    for _ in range(3):
        mpc.optimize(np.array([0.0]), np.array([1.0]))
    
    mpc.reset()
    np.testing.assert_array_almost_equal(mpc.u_seq, np.zeros_like(mpc.u_seq))

def test_mpc_set_constraints():
    f = lambda x, u, dt: x + u * dt
    mpc = ModelPredictiveControl(model_func=f, x0=np.zeros(1), horizon=3, dt=0.1, Q=np.eye(1), R=np.eye(1), u_min=-1, u_max=1)
    mpc.set_constraints(-2.0, 2.0)
    u_opt = mpc.optimize(np.array([0.0]), np.array([10.0]))
    assert np.all(u_opt >= -2.0)
    assert np.all(u_opt <= 2.0)

def test_mpc_set_weights():
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [0.1]])
    mpc = ModelPredictiveControl(model_func=None, x0=np.zeros(2), horizon=3, dt=0.1, Q=np.eye(2), R=np.eye(1), u_min=-1, u_max=1, A=A, B=B)
    new_Q = np.eye(2) * 5.0
    mpc.set_weights(new_Q, np.eye(1))
    np.testing.assert_array_almost_equal(mpc.Q, new_Q)
