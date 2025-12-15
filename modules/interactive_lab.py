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
    Real-time interactive simulation environment.

    Current scope:
    - Fixed-step state propagation
    - DC motor only
    - No visuals, no input, no controllers
    """

    def __init__(self, system_descriptor, params, dt=0.01):
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
        Initialize lab state and system-specific parameters.
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
        Advance simulation by one fixed timestep.

        Args:
            u: control input (voltage for DC motor)
            disturbance: external disturbance (e.g. load torque)
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
        self.state = None
        self.time = 0.0
        self.running = False

    def evaluate_rules(self):
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
        if self.control_mode == "MANUAL":
            return self.manual_input

        elif self.control_mode == "AUTO":
            if self.controller is None:
                raise RuntimeError("AUTO mode selected but no controller provided.")
            return self.controller(self.state_est, self.time)

        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")

    def set_manual_input(self, u):
        self.manual_input = float(u)

    def set_auto_controller(self, controller_fn):
        self.controller = controller_fn
        self.control_mode = "AUTO"

    def set_manual_mode(self):
        self.control_mode = "MANUAL"

    def handle_keyboard_input(self):
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
        self.fig, self.ax = plt.subplots()
        self.ax.axhline(0.0, color="k", linestyle="--")
        # self.ax.set_ylim(-1.0, 1.0)

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
        self.estimator = estimator
        self.use_estimator = True
        self.measurement_func = measurement_func


def simple_dc_motor_pid(omega_ref, Kp=1.0):
    def controller(state, t):
        omega = state[0]
        return Kp * (omega_ref - omega)

    return controller


def pendulum_lqr_controller(K):
    def controller(state, t):
        return float(-K @ state)

    return controller
