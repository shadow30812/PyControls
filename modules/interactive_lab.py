import select
import sys

import matplotlib.pyplot as plt
import numpy as np

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

    def __init__(self, system_descriptor, params, dt=0.01):
        """
        Initializes the simulation environment.

        Args:
            system_descriptor (SystemDescriptor): Metadata for the active system.
            params (dict): Physical parameters for the system dynamics.
            dt (float, optional): Simulation time step in seconds. Defaults to 0.01.
        """
        self.status = "RUNNING"
        self.failure_reason = None
        self.max_time = 20.0

        self.descriptor = system_descriptor
        self.params = params
        self.dt = dt
        self.system_instance = None

        self.state = None
        self.time = 0.0
        self.running = False

        self.control_mode = "MANUAL"
        self.controller = None
        self.manual_input = 0.0

        self.estimator = None  # type: ignore
        self.use_estimator = False
        self.state_est = None
        self.last_u = 0.0

    def initialize(self):
        """
        Resets the simulation state and configures system-specific goals.

        - DC Motor: Starts at rest. Goal is to reach 'omega_ref' within tolerance.
        - Pendulum: Starts slightly tilted (0.05 rad). Goal is to keep theta within limits.

        Raises:
            NotImplementedError: If the system ID is not recognized.
        """
        self.time = 0.0
        self.success_timer = 0.0
        self.running = True
        self.status = "RUNNING"
        self.failure_reason = None

        if self.descriptor.system_id == "dc_motor":
            self.state = np.array([0.0, 0.0], dtype=float)
            self.omega_ref = self.params.get("omega_ref", 1.0)
            self.omega_tol = self.params.get("omega_tol", 0.05)
            self.success_time_required = 3.0

        elif self.descriptor.system_id == "pendulum":
            self.state = np.array([0.0, 0.0, 0.05, 0.0], dtype=float)
            self.theta_limit = self.params.get("theta_limit", 0.5)
            self.success_time_required = 5.0

        else:
            raise NotImplementedError(
                f"Interactive lab not implemented for {self.descriptor.system_id}"
            )

        self.state_est = self.state.copy()

    def step(self, disturbance=0.0):
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
        if self.time >= self.max_time:
            self.status = "FAILED"
            self.failure_reason = "Time limit reached"
            return self.state

        if self.control_mode == "MANUAL":
            self.handle_keyboard_input()

        if self.status in ("SUCCESS", "FAILED"):
            return self.state

        if not self.running:
            raise RuntimeError("InteractiveLab.step() called before initialize().")

        u = self.get_control_input()
        self.last_u = u

        if self.descriptor.system_id == "dc_motor":
            dynamics = dc_motor_dynamics
        elif self.descriptor.system_id == "pendulum":
            dynamics = pendulum_dynamics
        else:
            raise NotImplementedError(
                f"Step not implemented for {self.descriptor.system_id}"
            )

        self.state = rk4_fixed_step(
            dynamics,
            self.state,
            u,
            self.dt,
            self.params,
            disturbance,
        )

        if self.use_estimator and self.estimator is not None:
            if self.measurement_func is not None:
                y = self.measurement_func(self.state).reshape(-1, 1)
            else:
                y = self.state.copy().reshape(-1, 1)
            self.estimator.predict(self.last_u, self.dt)
            self.estimator.update(y)
            self.state_est = self.estimator.x_hat.flatten()
        else:
            self.state_est = self.state.copy()

        self.time += self.dt
        self.evaluate_rules()

        if self.status == "SUCCESS" and self.control_mode == "AUTO":
            self.running = False

        return self.state

    def reset(self):
        """
        Clears the current simulation state, stopping execution.
        """
        self.state = None
        self.time = 0.0
        self.running = False

    def evaluate_rules(self):
        """
        Checks system-specific rules to determine SUCCESS or FAILED status.

        - DC Motor: Success if speed stays within tolerance of reference for 3 seconds.
        - Pendulum: Fail if angle exceeds limits. Success if it stays upright for 5 seconds.
        """
        if self.descriptor.system_id == "dc_motor":
            omega = self.state[0]
            error = abs(omega - self.omega_ref)

            if error <= self.omega_tol:
                self.success_timer += self.dt
                if self.success_timer >= self.success_time_required:
                    self.status = "SUCCESS"
            else:
                self.success_timer = 0.0

        elif self.descriptor.system_id == "pendulum":
            theta = self.state[2]

            if abs(theta) > self.theta_limit:
                self.status = "FAILED"
                self.failure_reason = "Pendulum fell"

            else:
                self.success_timer += self.dt
                if self.success_timer >= self.success_time_required:
                    self.status = "SUCCESS"

    def get_control_input(self):
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
            return self.controller(self.state_est, self.time)

        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")

    def set_manual_input(self, u):
        """
        Sets the control input value directly (used by manual interface).
        """
        self.manual_input = float(u)

    def set_auto_controller(self, controller_fn):
        """
        Registers an automatic controller function and switches to AUTO mode.

        Args:
            controller_fn (callable): A function f(state, time) -> u.
        """
        self.controller = controller_fn
        self.control_mode = "AUTO"

    def set_manual_mode(self):
        """
        Switches control authority to the manual user inputs.
        """
        self.control_mode = "MANUAL"

    def handle_keyboard_input(self):
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

        key = sys.stdin.read(1)

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

    def init_visualization(self):
        """
        Sets up the Matplotlib figure for real-time plotting.
        """
        self.fig, self.ax = plt.subplots()
        self.ax.axhline(0.0, color="k", linestyle="--")

        if self.descriptor.system_id == "dc_motor":
            self.ax.set_title("DC Motor Speed")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("ω (rad/s)")
            self.times = []
            self.values = []
            (self.line,) = self.ax.plot([], [], lw=2)

        elif self.descriptor.system_id == "pendulum":
            self.ax.set_title("Inverted Pendulum Angle")
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("θ (rad)")
            self.times = []
            self.values = []
            (self.line,) = self.ax.plot([], [], lw=2)

        plt.ion()
        plt.show()

    def update_visualization(self):
        """
        Updates the real-time plot with the latest simulation data.
        """
        self.times.append(self.time)

        if self.descriptor.system_id == "dc_motor":
            self.values.append(self.state[0])
        elif self.descriptor.system_id == "pendulum":
            self.values.append(self.state[2])

        self.line.set_data(self.times, self.values)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.001)

    def set_estimator(self, estimator, measurement_func=None):
        """
        Attaches a state estimator (e.g., Kalman Filter) to the simulation loop.

        Args:
            estimator: An object with predict(u, dt) and update(y) methods.
            measurement_func (callable, optional): Function mapping full state x to measurement y.
                                                   If None, assumes direct full state measurement.
        """
        self.estimator = estimator
        self.use_estimator = True
        self.measurement_func = measurement_func


def simple_dc_motor_pid(omega_ref, Kp=1.0):
    """
    Factory for a simple Proportional controller for the DC motor.
    """

    def controller(state, t):
        omega = state[0]
        return Kp * (omega_ref - omega)

    return controller


def pendulum_lqr_controller(K):
    """
    Factory for an LQR controller for the pendulum.
    Assumes u = -K * x.
    """

    def controller(state, t):
        return float(-K @ state)

    return controller
