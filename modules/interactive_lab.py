import select
import sys
from collections import deque
from typing import Any, Callable, Deque, Dict, Final, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike, NDArray

from core.control_utils import PIDController
from helpers.config import BATTERY_PID, THERMISTOR_PID
from helpers.system_registry import SystemDescriptor
from modules.physics_engine import (
    dc_motor_dynamics,
    pendulum_dynamics,
    rk4_fixed_step,
)


class InteractiveLab:
    """
    Manages a real-time interactive simulation environment.

    This class orchestrates the simulation loop, including:
    - Fixed-step state propagation using RK4 integration.
    - Handling user input (manual control) and switching to automatic controllers.
    - evaluating success/failure criteria (e.g., stability limits).
    - Optional state estimation and real-time visualization.

    Current support includes DC Motor (Speed Control) and Inverted Pendulum (Stabilization).
    """

    def __init__(
        self,
        system_descriptor: SystemDescriptor,
        params: Dict[str, Any],
        dt: float = 0.01,
    ) -> None:
        """
        Initializes the simulation environment.

        Args:
            system_descriptor (SystemDescriptor): Metadata for the active system.
            params (dict): Physical parameters for the system dynamics.
            dt (float, optional): Simulation time step in seconds. Defaults to 0.01.
        """
        self.status: str = "RUNNING"
        self.failure_reason: Optional[str] = None
        self.max_time: Final[float] = 20.0

        self.descriptor: Final[SystemDescriptor] = system_descriptor
        self.params: Final[Dict[str, Any]] = params
        self.dt: Final[float] = dt
        self.system_instance: Any = None

        self.state: Optional[NDArray[np.float64]] = None
        self.time: float = 0.0
        self.running: bool = False

        self.control_mode: str = "MANUAL"
        self.controller: Optional[Callable[[NDArray[np.float64], float], float]] = None
        self.manual_input: float = 0.0

        self.estimator: Any = None
        self.use_estimator: bool = False
        self.state_est: Optional[NDArray[np.float64]] = None

        self.last_u: float = 0.0
        self.last_avg: float = 0.0

    def initialize(self) -> None:
        """
        Resets the simulation state and configures system-specific goals.

        - DC Motor: Starts at rest. Goal is to reach 'omega_ref' within tolerance.
        - Pendulum: Starts slightly tilted (0.05 rad). Goal is to keep theta within limits.

        Raises:
            NotImplementedError: If the system ID is not recognized.
        """
        self.time = 0.0
        self.success_timer: float = 0.0
        self.running = True
        self.status = "RUNNING"
        self.failure_reason = None

        sid: str = self.descriptor.system_id

        if sid == "dc_motor":
            self.state = np.array([0.0, 0.0], dtype=float)
            self.omega_ref: float = self.params.get("omega_ref", 1.0)
            self.omega_tol: float = self.params.get("omega_tol", 0.05)
            self.success_time_required: float = 5.0
            self._dynamics = dc_motor_dynamics

        elif sid == "pendulum":
            self.state = np.array([0.0, 0.0, 0.05, 0.0], dtype=float)
            self.theta_limit: float = self.params.get("theta_limit", 0.5)
            self.success_time_required = 5.0
            self._dynamics: Callable[..., NDArray[np.float64]] = pendulum_dynamics

        elif sid == "thermistor":
            self.system_instance = self.descriptor.system_class(**self.params)
            self.system_instance.connect()
            self.state = np.array([25.0])
            cfg: Dict[str, Any] = THERMISTOR_PID

            tmp_pid = PIDController(
                Kp=cfg["Kp"],
                Ki=cfg["Ki"],
                Kd=cfg["Kd"],
                derivative_on_measurement=False,
                output_limits=cfg["output_limits"],
                integral_limits=cfg["integral_limits"],
            )
            self.set_auto_controller(
                lambda s, t: tmp_pid.update(s[0], self.params["setpoint"], self.dt)
            )
            self.control_mode = "AUTO"

        elif sid == "battery":
            self.system_instance = self.descriptor.system_class(**self.params)
            self.system_instance.connect()
            self.state = np.array([0.0])
            self.success_window: Deque[float] = deque()
            self.success_window_duration: float = 5.0
            self.success_tol: float = 0.05

            pwm_pid = PIDController(
                BATTERY_PID["Kp"],
                BATTERY_PID["Ki"],
                BATTERY_PID["Kd"],
                output_limits=(0, 255),
            )
            self.set_auto_controller(
                lambda s, t: pwm_pid.update(s[0], self.params["setpoint"], self.dt)
            )
            self.control_mode = "AUTO"

        else:
            raise NotImplementedError(
                f"Interactive lab not implemented for {self.descriptor.system_id}"
            )

        self.state_est = self.state.copy()

    def step(self, disturbance: float = 0.0) -> NDArray[np.float64]:
        """
        Advances the simulation by one fixed time step.

        This method:
        1. Checks time limits and input modes.
        2. Computes the control input (Manual vs Auto).
        3. Integrates the system dynamics using RK4.
        4. Updates the State Estimator (if configured) with the new measurement.
        5. Evaluates pass/fail rules.

        Args:
            disturbance (float, optional): External disturbance to apply. Defaults to 0.0.

        Returns:
            np.ndarray: The updated state vector.
        """
        if self.state is None:
            return np.zeros(0)

        if self.status in ("SUCCESS", "FAILED"):
            return self.state

        if not self.running:
            raise RuntimeError("InteractiveLab.step() called before initialize().")

        if self.control_mode == "MANUAL":
            self.handle_keyboard_input()

        u: float = self.get_control_input()
        self.last_u = u

        if self.descriptor.is_hardware and self.descriptor.system_id == "thermistor":
            self.system_instance.write_pwm(u)
            self.state = np.array([self.system_instance.read_temp()])

        elif self.descriptor.is_hardware and self.descriptor.system_id == "battery":
            self.system_instance.write_pwm(u)
            self.state = np.array([self.system_instance.read_voltage()])

        else:
            self.state = rk4_fixed_step(
                self._dynamics,
                self.state,
                u,
                self.dt,
                self.params,
                disturbance,
            )

        if self.use_estimator and self.estimator is not None:
            y: NDArray[np.float64] = (
                self.measurement_func(self.state)
                if self.measurement_func
                else self.state
            ).reshape(-1, 1)
            self.estimator.predict(self.last_u, self.dt)
            self.estimator.update(y)
            self.state_est = self.estimator.x_hat.flatten()
        else:
            self.state_est = self.state.copy()

        self.time += self.dt
        self.evaluate_rules()
        return self.state

    def reset(self) -> None:
        """
        Clears the current simulation state, stopping execution.
        Also stops hardware circuit(s) connected to the program.
        """
        if (
            self.descriptor.system_id in ("thermistor", "battery")
            and self.system_instance
        ):
            self.system_instance.close()

        self.state = None
        self.time = 0.0
        self.running = False

    def evaluate_rules(self) -> None:
        """
        Checks system-specific rules to determine SUCCESS or FAILED status.

        - DC Motor: Success if speed stays within tolerance of reference for 3 seconds.
        - Pendulum: Fail if angle exceeds limits. Success if it stays upright for 5 seconds.
        """
        if self.descriptor.system_id == "dc_motor":
            if self.state is not None:
                omega: float = self.state[0]
            else:
                raise RuntimeError
            error: float = abs(omega - self.omega_ref)

            if error <= self.omega_tol:
                self.success_timer += self.dt
                if self.success_timer >= self.success_time_required:
                    self.status = "SUCCESS"
            else:
                self.success_timer = 0.0

        elif self.descriptor.system_id == "pendulum":
            if self.state is not None:
                theta: float = self.state[2]
            else:
                raise RuntimeError

            if abs(theta) > self.theta_limit:
                self.status = "FAILED"
                self.failure_reason = "Pendulum fell"

            else:
                self.success_timer += self.dt
                if self.success_timer >= self.success_time_required:
                    self.status = "SUCCESS"

        elif self.descriptor.system_id == "thermistor":
            if self.state is not None:
                temp: float = self.state[0]
            else:
                raise RuntimeError
            if abs(temp - self.params["setpoint"]) <= 1.0:
                self.success_timer += self.dt
                if self.success_timer >= 5.0:
                    self.status = "SUCCESS"
            else:
                self.success_timer = 0.0

        elif self.descriptor.system_id == "battery":
            if self.state_est is not None:
                v: float = self.state_est[0]
            else:
                raise RuntimeError
            self.success_window.append(v)
            max_len: int = int(self.success_window_duration / self.dt)
            avg_v: float = sum(self.success_window) / len(self.success_window)
            self.last_avg = avg_v

            while len(self.success_window) > max_len:
                self.success_window.popleft()

            if len(self.success_window) == max_len:
                if abs(avg_v - self.params["setpoint"]) <= self.success_tol:
                    self.status = "SUCCESS"

    def get_control_input(self) -> float:
        """
        Retrieves the control signal 'u' based on the active mode.

        Returns:
            float: The control input value.
        """
        if self.control_mode == "MANUAL":
            return self.manual_input

        elif self.control_mode == "AUTO":
            if self.controller is None:
                raise RuntimeError("AUTO mode selected but no controller provided.")
            if self.state_est is not None:
                return self.controller(self.state_est, self.time)
            else:
                raise RuntimeError

        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")

    def set_manual_input(self, u: Union[float, int]) -> None:
        """
        Sets the control input value directly (used by manual interface).
        """
        self.manual_input = float(u)

    def set_auto_controller(
        self, controller_fn: Callable[[NDArray[np.float64], float], float]
    ) -> None:
        """
        Registers an automatic controller function and switches to AUTO mode.

        Args:
            controller_fn (callable): A function f(state, time) -> u.
        """
        self.controller = controller_fn
        self.control_mode = "AUTO"

    def set_manual_mode(self) -> None:
        """
        Switches control authority to the manual user inputs.
        """
        self.control_mode = "MANUAL"

    def handle_keyboard_input(self) -> None:
        """
        Poller for non-blocking keyboard input (Unix-style).

        Controls:
            'a'/'d': Decrease/Increase input.
            's': Zero input.
            'm': Switch to Manual mode.
            'o': Switch to Auto mode.
            'q': Quit simulation (Fail).
        """
        if not sys.stdin.isatty():
            return

        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return

        key: str = sys.stdin.read(1)

        if key == "q":
            self.status = "FAILED"
            self.failure_reason = "User quit"
            return

        if self.control_mode != "MANUAL":
            return

        if key == "a":
            self.manual_input -= 0.5
        elif key == "d":
            self.manual_input += 0.5
        elif key == "s":
            self.manual_input = 0.0
        elif key == "m":
            self.set_manual_mode()
        elif key == "o":
            if self.controller is not None:
                self.control_mode = "AUTO"

    def init_visualization(self) -> None:
        """
        Sets up the Matplotlib figure for real-time plotting.
        """

        if self.descriptor.system_id == "dc_motor":
            Fax: Tuple[Figure, Axes] = plt.subplots()
            self.fig, self.ax = Fax
            self.ax.set_title("DC Motor Speed")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("ω (rad/s)")
            self.ax.axhline(0.0, color="k", linestyle="--")
            self.times: List[float] = []
            self.values: List[float] = []
            self.line: Line2D
            (self.line,) = self.ax.plot([], [], lw=2)

        elif self.descriptor.system_id == "pendulum":
            Fax: Tuple[Figure, Axes] = plt.subplots()
            self.fig, self.ax = Fax
            self.ax.set_title("Inverted Pendulum Angle")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("θ (rad)")
            self.ax.axhline(0.0, color="k", linestyle="--")
            self.times = []
            self.values = []
            (self.line,) = self.ax.plot([], [], lw=2)

        elif self.descriptor.system_id == "thermistor":
            Fax: Tuple[Figure, Axes] = plt.subplots()
            self.fig, self.ax = Fax
            self.ax.set_title("HIL Temperature Control")
            self.ax.set_ylabel("Temp (°C)")
            self.ax.axhline(
                self.params["setpoint"], color="g", ls="--", label="Setpoint"
            )
            self.ax.legend()
            self.times = []
            self.values = []
            (self.line,) = self.ax.plot([], [], "b-")

        if self.descriptor.system_id == "battery":
            Faxes: Tuple[Figure, NDArray[Any]] = plt.subplots(2, 2, figsize=(12, 8))
            self.fig, self.axes = Faxes

            self.ax_v: Axes = self.axes[0, 0]
            self.ax_u: Axes = self.axes[0, 1]
            self.ax_inn: Axes = self.axes[1, 0]
            self.ax_p: Axes = self.axes[1, 1]

            self.ax_v.set_title("Voltage Tracking (V)")
            self.ax_u.set_title("Control Effort (PWM)")
            self.ax_inn.set_title("Innovation (Residue)")
            self.ax_p.set_title("Filter Covariance (P)")

            self.line_v_est: Line2D
            self.line_v_raw: Line2D
            self.line_u: Line2D
            self.line_inn: Line2D
            self.line_p: Line2D

            (self.line_v_est,) = self.ax_v.plot([], [], "b-", label="KF Estimate")
            (self.line_v_raw,) = self.ax_v.plot(
                [], [], "r.", alpha=0.2, label="Raw ADC"
            )
            self.ax_v.axhline(
                self.params["setpoint"], color="k", ls="--", label="Target"
            )
            self.ax_v.legend(loc="upper right", fontsize="small")

            (self.line_u,) = self.ax_u.plot([], [], "g-", label="PWM Input")
            (self.line_inn,) = self.ax_inn.plot([], [], "m-", label="Innovation")
            (self.line_p,) = self.ax_p.plot([], [], "k-", label="P (Certainty)")

            self.hist_v_raw: List[float] = []
            self.hist_inn: List[float] = []
            self.hist_p: List[float] = []
            self.times = []
            self.hist_v_est: List[float] = []
            self.hist_u: List[float] = []

            plt.tight_layout()

        plt.ion()
        plt.show()

    def update_visualization(self) -> None:
        """
        Updates the real-time plot with the latest simulation data.
        """
        self.times.append(self.time)
        if self.status == "SUCCESS":
            plt.savefig(f"final_success_plot_{self.descriptor.system_id}.png")

        if self.descriptor.system_id == "dc_motor" and self.state is not None:
            self.values.append(self.state[0])
        elif self.descriptor.system_id == "pendulum" and self.state is not None:
            self.values.append(self.state[2])
        elif self.descriptor.system_id == "thermistor" and self.state is not None:
            self.values.append(self.state[0])

        elif (
            self.descriptor.system_id == "battery"
            and self.state is not None
            and self.state_est is not None
        ):
            self.hist_v_est.append(self.state_est[0])
            self.hist_v_raw.append(self.state[0])
            self.hist_u.append(self.last_u)

            if self.estimator:
                innovation: float = self.hist_v_raw[-1] - self.hist_v_est[-1]
                self.hist_inn.append(innovation)
                self.hist_p.append(self.estimator.P[0, 0])

            self.line_v_est.set_data(self.times, self.hist_v_est)
            self.line_v_raw.set_data(self.times, self.hist_v_raw)
            self.line_u.set_data(self.times, self.hist_u)
            self.line_inn.set_data(self.times, self.hist_inn)
            self.line_p.set_data(self.times, self.hist_p)

            ax: Axes
            for ax in self.axes.flat:
                ax.relim()
                ax.autoscale_view()

            plt.pause(0.001)
            return

        self.line.set_data(self.times, self.values)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.001)

    def set_estimator(
        self,
        estimator: Any,
        measurement_func: Optional[
            Callable[[NDArray[np.float64]], NDArray[np.float64]]
        ] = None,
    ) -> None:
        """
        Attaches a state estimator (e.g., Kalman Filter) to the simulation loop.

        Args:
            estimator: An object with predict(u, dt) and update(y) methods.
            measurement_func (callable, optional): Function mapping full state x to measurement y.
                                                   If None, assumes direct full state measurement.
        """
        self.estimator = estimator
        self.use_estimator = True
        self.measurement_func: Optional[
            Callable[[NDArray[np.float64]], NDArray[np.float64]]
        ] = measurement_func


def simple_dc_motor_pid(
    omega_ref: float, Kp: float = 1.0
) -> Callable[[NDArray[np.float64], float], float]:
    """
    Factory for a simple Proportional controller for the DC motor.
    """

    def controller(state: NDArray[np.float64], t: float) -> float:
        omega: float = state[0]
        return Kp * (omega_ref - omega)

    return controller


def pendulum_lqr_controller(
    K: NDArray[np.float64],
) -> Callable[[NDArray[np.float64], float], float]:
    """
    Factory for an LQR controller for the pendulum.
    Assumes u = -K * x.
    """

    def controller(state: NDArray[np.float64], t: float) -> float:
        return float(-K @ state)

    return controller
