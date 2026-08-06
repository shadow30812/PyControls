import pytest
import numpy as np
from interfaces.manager import ControllerManager
from core.control_utils import PIDController
from core.estimator import KalmanFilter

def test_manager_register_get():
    mgr = ControllerManager()
    pid = PIDController(1.0, 0.1, 0.01)
    handle = mgr.register(pid, "my_pid")
    assert handle == "my_pid"
    assert mgr.get(handle) is pid

def test_manager_auto_naming():
    mgr = ControllerManager()
    pid = PIDController(1.0, 0.1, 0.01)
    handle = mgr.register(pid)
    assert "PIDController" in handle

def test_manager_duplicate_handle():
    mgr = ControllerManager()
    pid = PIDController(1.0, 0.1, 0.01)
    mgr.register(pid, "pid1")
    with pytest.raises(KeyError):
        mgr.register(pid, "pid1")

def test_manager_destroy():
    mgr = ControllerManager()
    pid = PIDController(1.0, 0.1, 0.01)
    handle = mgr.register(pid, "pid1")
    mgr.destroy(handle)
    with pytest.raises(KeyError):
        mgr.get(handle)

def test_manager_destroy_all():
    mgr = ControllerManager()
    mgr.register(PIDController(1.0, 0.1, 0.0), "p1")
    mgr.register(PIDController(1.0, 0.1, 0.0), "p2")
    mgr.register(PIDController(1.0, 0.1, 0.0), "p3")
    mgr.destroy_all()
    assert len(mgr.list_handles()) == 0

def test_manager_list_handles():
    mgr = ControllerManager()
    mgr.register(PIDController(1.0, 0.1, 0.0), "p1")
    handles = mgr.list_handles()
    assert "p1" in handles
    assert "PIDController" in handles["p1"]

def test_manager_step_estimator():
    mgr = ControllerManager()
    A = np.array([[1, 0.1], [0, 1]])
    B = np.array([[0], [0.1]])
    C = np.array([[1, 0]])
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    x0 = np.array([0.0, 0.0])
    
    kf = KalmanFilter(A, B, C, Q, R, x0)
    handle = mgr.register(kf, "kf1")
    old_state = kf.get_state().copy()
    
    mgr.step_estimator(handle, np.array([1.0]), np.array([0.5]), 0.1)
    assert not np.allclose(kf.get_state(), old_state)

def test_manager_step_controller():
    mgr = ControllerManager()
    pid = PIDController(1.0, 0.1, 0.01)
    handle = mgr.register(pid, "pid1")
    u = mgr.step_controller(handle, np.array([1.0]), np.array([2.0]), 0.01)
    assert isinstance(u, np.ndarray)

def test_manager_save_load_all():
    mgr = ControllerManager()
    pid = PIDController(1.0, 0.1, 0.01)
    mgr.register(pid, "pid1")
    
    mgr.step_controller("pid1", np.array([1.0]), np.array([2.0]), 0.01)
    
    states = mgr.save_all()
    mgr.reset("pid1")
    mgr.load_all(states)
    
    pid_restored = mgr.get("pid1")
    assert pid_restored.integral_error > 0

def test_manager_namespaces():
    mgr = ControllerManager()
    pid1 = PIDController(1.0, 0.1, 0.01)
    pid2 = PIDController(1.0, 0.1, 0.01)
    
    mgr.register(pid1, "drone_1/pid")
    mgr.register(pid2, "drone_2/pid")
    
    assert mgr.get("drone_1/pid") is pid1
    assert mgr.get("drone_2/pid") is pid2
