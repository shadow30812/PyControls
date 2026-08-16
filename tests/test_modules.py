import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from typing import Dict, Type

from numpy.typing import NDArray

from core.exceptions import PyControlsError
from helpers.system_registry import SystemDescriptor
from modules.interactive_lab import InteractiveLab
from modules.physics_engine import (
    dc_motor_dynamics,
    pendulum_dynamics,
    rk4_fixed_step,
)


class DummyDescriptor(SystemDescriptor):
    """Minimal system descriptor for InteractiveLab tests."""

    def __init__(self, system_id: str, is_hardware: bool = False) -> None:
        self.system_id: str = system_id
        self.is_hardware: bool = is_hardware


class DummyEstimator:
    """Minimal estimator stub to verify estimator hooks."""

    def __init__(self, n: int) -> None:
        self.x_hat: NDArray[np.float64] = np.zeros((n, 1))
        self.predict_calls: int = 0
        self.update_calls: int = 0

    def predict(self, u: float, dt: float) -> None:
        self.predict_calls += 1

    def update(self, y: NDArray[np.float64]) -> None:
        self.update_calls += 1
        self.x_hat = y.copy()


class TestPhysicsEngine(unittest.TestCase):
    """
    Exhaustive tests for physics_engine guarantees.
    """

    def test_dc_motor_dynamics_shape_and_finiteness(self) -> None:
        params = {"J": 0.01, "b": 0.1, "K": 0.01, "R": 1.0, "L": 0.5}
        x = np.array([0.0, 0.0])
        dx = dc_motor_dynamics(0.0, x, 1.0, params)

        self.assertEqual(dx.shape, (2,))
        self.assertTrue(np.all(np.isfinite(dx)))

    def test_dc_motor_zero_input_decay(self) -> None:
        params = {"J": 0.01, "b": 0.1, "K": 0.01, "R": 1.0, "L": 0.5}
        x = np.array([1.0, 0.0])
        dx = dc_motor_dynamics(0.0, x, 0.0, params)

        self.assertLess(dx[0], 0.0)

    def test_pendulum_dynamics_shape_and_finiteness(self) -> None:
        params = {"M": 1.0, "m": 0.1, "l": 1.0}
        x = np.array([0.0, 0.0, 0.1, 0.0])
        dx = pendulum_dynamics(0.0, x, 0.0, params)

        self.assertEqual(dx.shape, (4,))
        self.assertTrue(np.all(np.isfinite(dx)))

    def test_rk4_constant_derivative(self) -> None:
        def constant_dyn(
            t: float,
            x: NDArray[np.float64],
            u: float,
            params: Dict[str, float],
            disturbance: float = 0.0,
        ) -> NDArray[np.float64]:
            return np.ones_like(x)

        x0 = np.zeros(3)
        dt = 0.1
        x1 = rk4_fixed_step(constant_dyn, x0, 0.0, dt, {})

        np.testing.assert_allclose(x1, x0 + dt, rtol=1e-12)

    def test_rk4_does_not_mutate_input(self) -> None:
        def dyn(
            t: float,
            x: NDArray[np.float64],
            u: float,
            params: Dict[str, float],
            disturbance: float = 0.0,
        ) -> NDArray[np.float64]:
            return -x

        x0 = np.array([1.0, 2.0])
        x0_copy = x0.copy()

        rk4_fixed_step(dyn, x0, 0.0, 0.1, {})
        np.testing.assert_array_equal(x0, x0_copy)


class TestInteractiveLab(unittest.TestCase):
    """
    Exhaustive tests for InteractiveLab contractual behavior.
    """

    def test_initialize_dc_motor(self) -> None:
        desc = DummyDescriptor("dc_motor")
        params = {"J": 0.01, "b": 0.1, "K": 0.01, "R": 1.0, "L": 0.5}

        lab = InteractiveLab(desc, params)
        lab.initialize()

        assert lab.state is not None
        self.assertTrue(lab.running)
        self.assertEqual(lab.state.shape, (2,))
        self.assertEqual(lab.status, "RUNNING")

    def test_initialize_pendulum(self) -> None:
        desc = DummyDescriptor("pendulum")
        params = {"M": 1.0, "m": 0.1, "l": 1.0}

        lab = InteractiveLab(desc, params)
        lab.initialize()

        assert lab.state is not None
        self.assertEqual(lab.state.shape, (4,))
        self.assertTrue(lab.running)

    def test_initialize_invalid_system(self) -> None:
        desc = DummyDescriptor("nonsense")
        lab = InteractiveLab(desc, {})

        with self.assertRaises(NotImplementedError):
            lab.initialize()

    def test_step_requires_initialize(self) -> None:
        desc = DummyDescriptor("dc_motor")
        lab = InteractiveLab(desc, {})
        with self.assertRaises(RuntimeError):
            lab.step()

    def test_manual_step_advances_time(self) -> None:
        desc = DummyDescriptor("dc_motor")
        params = {"J": 0.01, "b": 0.1, "K": 0.01, "R": 1.0, "L": 0.5}

        lab = InteractiveLab(desc, params, dt=0.05)
        lab.initialize()
        t0 = lab.time
        lab.step()

        self.assertAlmostEqual(lab.time, t0 + 0.05)

    def test_auto_controller_called(self) -> None:
        desc = DummyDescriptor("dc_motor")
        params = {"J": 0.01, "b": 0.1, "K": 0.01, "R": 1.0, "L": 0.5}

        calls = {"n": 0}

        def controller(state: NDArray[np.float64], t: float) -> float:
            calls["n"] += 1
            return 0.0

        lab = InteractiveLab(desc, params)
        lab.initialize()
        lab.set_auto_controller(controller)
        lab.step()

        self.assertGreater(calls["n"], 0)

    def test_auto_mode_without_controller_raises(self) -> None:
        desc = DummyDescriptor("dc_motor")
        lab = InteractiveLab(desc, {})
        lab.initialize()
        lab.control_mode = "AUTO"

        with self.assertRaises(RuntimeError):
            lab.get_control_input()

    def test_estimator_hooks_called(self) -> None:
        desc = DummyDescriptor("dc_motor")
        params = {"J": 0.01, "b": 0.1, "K": 0.01, "R": 1.0, "L": 0.5}
        est = DummyEstimator(2)

        lab = InteractiveLab(desc, params)
        lab.initialize()
        lab.set_estimator(est)
        lab.step()

        assert lab.state_est is not None
        self.assertEqual(est.predict_calls, 1)
        self.assertEqual(est.update_calls, 1)
        self.assertEqual(lab.state_est.shape, (2,))

    def test_visualization_init_and_update(self) -> None:
        desc = DummyDescriptor("dc_motor")
        params = {"J": 0.01, "b": 0.1, "K": 0.01, "R": 1.0, "L": 0.5}

        lab = InteractiveLab(desc, params)
        lab.initialize()
        lab.init_visualization()

        self.assertTrue(hasattr(lab, "line"))
        lab.update_visualization()


class TestExceptions(unittest.TestCase):
    """
    Contract tests for custom exception hierarchy.
    """

    def test_exceptions_inherit_base(self) -> None:
        from core.exceptions import (
            ControllerConfigError,
            ConvergenceError,
            DimensionMismatchError,
            InvalidParameterError,
            SingularMatrixError,
            SolverError,
            UnstableSystemError,
        )

        exc: Type[PyControlsError]
        for exc in [
            DimensionMismatchError,
            SingularMatrixError,
            ConvergenceError,
            UnstableSystemError,
            InvalidParameterError,
            SolverError,
            ControllerConfigError,
        ]:
            self.assertTrue(issubclass(exc, PyControlsError))


class TestPhysicsEngineExtended(unittest.TestCase):
    """Additional non-fragile tests for physics_engine."""

    def test_dc_motor_dynamics_shape_and_finiteness(self) -> None:
        params = {"J": 0.01, "b": 0.1, "K": 0.01, "R": 1.0, "L": 0.5}
        x = np.array([0.1, 0.2])
        u = 1.0
        dx = dc_motor_dynamics(0.0, x, u, params)
        self.assertEqual(dx.shape, (2,))
        self.assertTrue(np.all(np.isfinite(dx)))

    def test_pendulum_dynamics_shape_and_finiteness(self) -> None:
        params = {"M": 1.0, "m": 0.1, "l": 0.5, "b": 0.05, "g": 9.81}
        x = np.array([0.0, 0.0, 0.1, 0.0])
        u = 0.0
        dx = pendulum_dynamics(0.0, x, u, params)
        self.assertEqual(dx.shape, (4,))
        self.assertTrue(np.all(np.isfinite(dx)))

    def test_rk4_fixed_step_identity_dynamics(self) -> None:
        def zero_dyn(t, x, u, params, disturbance=0.0):
            return np.zeros_like(x)

        x0 = np.array([1.0, -2.0, 3.0])
        x1 = rk4_fixed_step(zero_dyn, x0, u=0.0, dt=0.1, params={})
        np.testing.assert_array_equal(x1, x0)

    def test_rk4_determinism(self) -> None:
        params = {"J": 0.01, "b": 0.1, "K": 0.01, "R": 1.0, "L": 0.5}
        x = np.array([0.1, 0.2])
        u = 0.5
        x1 = rk4_fixed_step(dc_motor_dynamics, x, u, 0.01, params)
        x2 = rk4_fixed_step(dc_motor_dynamics, x, u, 0.01, params)
        np.testing.assert_allclose(x1, x2)


class TestInteractiveLabExtended(unittest.TestCase):
    """Additional non-fragile tests for InteractiveLab."""

    def _make_minimal_lab(self, system_id: str) -> InteractiveLab:
        class DummyDescriptor(SystemDescriptor):
            def __init__(self, sid: str) -> None:
                self.system_id: str = sid
                self.is_hardware: bool = False

        params = {"J": 1.0, "b": 0.1, "K": 0.1, "R": 1.0, "L": 1.0}

        lab = InteractiveLab(
            DummyDescriptor("dc_motor"),
            params=params,
        )
        lab.control_mode = "MANUAL"
        lab.handle_keyboard_input = lambda: None
        lab.get_control_input = lambda: 0.0
        lab.evaluate_rules = lambda: None
        return lab

    def test_initialize_sets_running(self) -> None:
        lab = self._make_minimal_lab("dc_motor")
        lab.initialize()
        self.assertTrue(lab.running)
        self.assertEqual(lab.status, "RUNNING")

    def test_step_advances_time(self) -> None:
        lab = self._make_minimal_lab("dc_motor")
        lab.initialize()
        t0 = lab.time
        lab.step()
        self.assertAlmostEqual(lab.time, t0 + lab.dt)

    def test_step_preserves_state_shape(self) -> None:
        lab = self._make_minimal_lab("pendulum")
        lab.initialize()
        assert lab.state is not None
        x0 = lab.state.copy()
        x1 = lab.step()
        self.assertEqual(x0.shape, x1.shape)

    def test_step_without_initialize_raises(self) -> None:
        lab = self._make_minimal_lab("dc_motor")
        with self.assertRaises(RuntimeError):
            lab.step()


if __name__ == "__main__":
    unittest.main()
