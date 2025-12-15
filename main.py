import importlib
import inspect
import os
import pkgutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
import config
from core.control_utils import PIDController
from core.ekf import ExtendedKalmanFilter
from core.estimator import KalmanFilter
from core.math_utils import make_system_func
from core.mpc import ModelPredictiveControl
from core.solver import ExactSolver, NonlinearSolver
from core.ukf import UnscentedKalmanFilter
from exit import flush, kill, stop
from modules.interactive_lab import InteractiveLab, pendulum_lqr_controller
from modules.physics_engine import pendulum_dynamics, rk4_fixed_step
from system_registry import SYSTEM_REGISTRY


def load_available_systems():
    """
    Dynamically discovers and loads system classes from the 'systems' package.

    Returns:
        dict: A dictionary mapping system names to their classes.
    """
    systems = {}
    systems_path = os.path.join(os.getcwd(), "systems")

    for _, name, _ in pkgutil.iter_modules([systems_path]):
        module_name = f"systems.{name}"
        try:
            module = importlib.import_module(module_name)
            for member_name, member_obj in inspect.getmembers(module, inspect.isclass):
                if (
                    hasattr(member_obj, "get_closed_loop_tf")
                    and hasattr(member_obj, "get_disturbance_tf")
                    and member_obj.__module__ == module_name
                ):
                    systems[member_name] = member_obj

        except Exception as e:
            print(f"Warning: Could not load system '{name}': {e}")

    return systems


class PyControlsApp:
    """
    Main Application Controller.
    Handles user interaction, menu navigation, and simulation orchestration.
    """

    def __init__(self):
        self.available_systems = load_available_systems()
        if not self.available_systems:
            print("Error: No valid system classes found in systems/ folder.")
            sys.exit(1)

        self.current_system_id = "dc_motor"
        self.current_descriptor = SYSTEM_REGISTRY[self.current_system_id]

        self.system_name = self.current_descriptor.display_name.replace(" ", "")
        self.SystemClass = self.current_descriptor.system_class

        if self.SystemClass is not None:
            temp_instance = self.SystemClass()
            self.active_params = temp_instance.params.copy()
        else:
            self.active_params = {}

        if self.current_system_id == "dc_motor" and hasattr(config, "MOTOR_PARAMS"):
            self.active_params.update(config.MOTOR_PARAMS)
        elif self.current_system_id == "pendulum" and hasattr(
            config, "PENDULUM_PARAMS"
        ):
            self.active_params.update(config.PENDULUM_PARAMS)

        self.system = (
            self.current_descriptor.system_class()
            if self.current_descriptor.system_class is not None
            else None
        )

        self.controllers = config.CONTROLLERS.copy()
        self.sim_params = config.SIM_PARAMS.copy()
        self.dist_params = config.DISTURBANCE_PARAMS.copy()
        self.running = True

    def clear_screen(self):
        """Clears the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self):
        """Prints the application banner and current system parameters."""
        print("\n" + "=" * 60)
        print(f"   PyControls Engineering Suite | System: {self.system_name}   ")
        print("=" * 60)
        param_list = [f"{k}={v}" for k, v in self.active_params.items()]
        param_str = ", ".join(param_list)
        if len(param_str) > 55:
            print(f"Params: {param_str[:55]}...")
        else:
            print(f"Params: {param_str}")
        print("-" * 60)

    def main_menu(self):
        """Displays the main menu loop."""
        self.clear_screen()
        while self.running:
            self.print_header()
            print("[1] Run Standard Simulation (Time-Domain Response)")
            print("[2] Run System Analysis (Frequency, Poles, Phase Plane)")
            print("[3] Edit System Parameters")
            print("[4] Edit Disturbance Settings")
            print("[5] Switch Active System")
            print("[6] Run Parameter Estimation Demo (EKF)")
            print("[7] Run Nonlinear State Est. Demo (UKF)")
            print("[8] Run Model Predictive Control Demo (MPC)")
            print("[9] Interactive Lab")
            print("[10] Run Custom Non-Linear Simulation")
            print("[q] Exit")

            choice = input("\nSelect Option: ").strip()

            if choice == "1":
                self.run_preset_dashboard()
            elif choice == "2":
                self.run_analysis_dashboard()
            elif choice == "3":
                self.edit_params_menu()
            elif choice == "4":
                self.edit_disturbance_menu()
            elif choice == "5":
                self.switch_system_menu()
            elif choice == "6":
                self.run_parameter_estimation()
            elif choice == "7":
                self.run_ukf_demo()
            elif choice == "8":
                self.run_mpc_demo()
            elif choice == "9":
                self.run_interactive_lab()
            elif choice == "10":
                self.run_custom_simulation()
            elif choice == "q":
                self.running = False
            else:
                input("Invalid option. Press Enter...")

    def switch_system_menu(self):
        """Menu to switch between available dynamic systems."""
        self.clear_screen()
        print("\nAvailable Systems:")

        system_ids = list(SYSTEM_REGISTRY.keys())
        for i, sys_id in enumerate(system_ids):
            desc = SYSTEM_REGISTRY[sys_id]
            print(f"[{i + 1}] {desc.display_name}")

        print("[b] Back")
        sel = input("\nSelect System: ").strip()

        if sel == "b":
            return

        try:
            idx = int(sel) - 1
            if 0 <= idx < len(system_ids):
                new_id = system_ids[idx]

                if new_id != self.current_system_id:
                    self.current_system_id = new_id
                    self.current_descriptor = SYSTEM_REGISTRY[new_id]

                    self.SystemClass = self.current_descriptor.system_class
                    self.system_name = self.current_descriptor.display_name.replace(
                        " ", ""
                    )

                    if self.SystemClass is not None:
                        self.system = self.SystemClass()
                        self.active_params = self.SystemClass().params.copy()
                        if new_id == "dc_motor" and hasattr(config, "MOTOR_PARAMS"):
                            self.active_params.update(config.MOTOR_PARAMS)
                        elif new_id == "pendulum" and hasattr(
                            config, "PENDULUM_PARAMS"
                        ):
                            self.active_params.update(config.PENDULUM_PARAMS)

                        self.system.params = self.active_params.copy()
                    else:
                        self.system = None
                        self.active_params = {}

                    print(f"\nSwitched to {self.current_descriptor.display_name}")
                    time.sleep(1)
                    self.clear_screen()
        except ValueError:
            pass

    def edit_params_menu(self):
        """Menu to edit the physical parameters of the current system."""
        print(f"\nCurrent Parameters for {self.system_name}:")
        for k, v in self.active_params.items():
            print(f"  [{k}] : {v}")
        key = input("\nEnter parameter key to edit (or 'b' to go back): ").strip()
        if key == "b":
            return
        if key in self.active_params:
            try:
                val = float(input(f"New value for {key}: "))
                self.active_params[key] = val
            except ValueError:
                print("Invalid number.")
        else:
            print("Unknown parameter.")

    def edit_disturbance_menu(self):
        """Menu to configure the external disturbance."""
        print(f"\n[1] Toggle Enable (Current: {self.dist_params['enabled']})")
        print(f"[2] Set Magnitude (Current: {self.dist_params['magnitude']})")
        print(f"[3] Set Time      (Current: {self.dist_params['time']})")
        print("[b] Back")
        choice = input("\nChoice: ").strip()
        if choice == "b":
            return
        if choice == "1":
            self.dist_params["enabled"] = not self.dist_params["enabled"]
        elif choice == "2":
            try:
                self.dist_params["magnitude"] = float(input("New Magnitude: "))
            except Exception:
                pass
        elif choice == "3":
            try:
                self.dist_params["time"] = float(input("New Time: "))
            except Exception:
                pass

    def simulate_preset_system(self, system_instance, ctrl_config):
        if not self.current_descriptor.supports_analysis:
            print("\nAnalysis & metrics are only available for linear systems.")
            input("Press Enter to return to menu...")
            return

        dt = self.sim_params["dt"]
        t_end = self.sim_params["t_end"]

        if hasattr(system_instance, "get_state_space"):
            ss_real = system_instance.get_state_space()
            solver_real = ExactSolver(ss_real.A, ss_real.B, ss_real.C, ss_real.D, dt)
        else:
            return np.array([]), np.array([]), np.array([])

        kf = None
        if hasattr(system_instance, "get_augmented_state_space"):
            ss_aug = system_instance.get_augmented_state_space()
            solver_aug_math = ExactSolver(ss_aug.A, ss_aug.B, ss_aug.C, ss_aug.D, dt)

            n_states = ss_aug.A.shape[0]
            Q = np.eye(n_states) * config.PRESET_SIM_PARAMS["kf_Q_scale"]
            Q[-1, -1] = config.PRESET_SIM_PARAMS[
                "kf_Q_dist_scale"
            ]  # Higher variance for disturbance state
            R = np.eye(ss_aug.C.shape[0]) * config.PRESET_SIM_PARAMS["kf_R_scale"]

            kf = KalmanFilter(
                solver_aug_math.Phi,
                solver_aug_math.Gamma,
                ss_aug.C,
                Q,
                R,
                x0=np.zeros(n_states),
            )

        use_lqr = self.current_system_id == "pendulum"
        lqr_K = None
        pid = None

        if use_lqr:
            if hasattr(system_instance, "dlqr_gain"):
                lqr_K = system_instance.dlqr_gain()
            else:
                print("Error: System does not support LQR.")
                return np.array([]), np.array([]), np.array([]), np.array([])
        else:
            pid = PIDController(
                Kp=ctrl_config["Kp"],
                Ki=ctrl_config["Ki"],
                Kd=ctrl_config["Kd"],
                derivative_on_measurement=False,
                output_limits=config.PRESET_SIM_PARAMS["pid_output_limits"],
                tau=config.PRESET_SIM_PARAMS["pid_tau"],
            )

        t_values = np.linspace(0, t_end, int(t_end / dt))
        y_real_hist = []
        x_est_hist = []
        u_hist = []

        # Initial State for Pendulum (must be non-zero to show stabilization)
        if use_lqr:
            solver_real.x = np.array(
                [[0.0], [0.0], [0.1], [0.0]]
            )  # Start at 0.1 rad tilt
            if kf:
                kf.x_hat[:4] = solver_real.x

        for t in t_values:
            dist_val = 0.0
            if self.dist_params["enabled"] and t >= self.dist_params["time"]:
                dist_val = self.dist_params["magnitude"]

            if self.current_system_id == "pendulum":
                ref_signal = self.sim_params["step_angle"] if t > 0 else 0
            elif self.current_system_id == "dc_motor":
                ref_signal = self.sim_params["step_volts"] if t > 0 else 0

            # Get Feedback
            if self.current_system_id == "pendulum":
                # LQR needs full state vector
                if kf:
                    # KF x_hat is (5,1) [x, dx, theta, dtheta, dist]
                    # LQR K is (1,4). We need first 4 states.
                    feedback_vec = kf.x_hat[:4]
                else:
                    feedback_vec = solver_real.x
            else:
                # PID needs scalar error
                x_idx = 0  # Speed
                if kf:
                    feedback = kf.x_hat[x_idx, 0]
                else:
                    feedback = solver_real.x[x_idx, 0]

            # Calculate Control
            if use_lqr:
                # u = -K * x
                u_val = -(lqr_K @ feedback_vec).item()
                # Clip for realism
                u_val = max(
                    config.PRESET_SIM_PARAMS["lqr_clip_min"],
                    min(config.PRESET_SIM_PARAMS["lqr_clip_max"], u_val),
                )
            else:
                u_val = pid.update(measurement=feedback, setpoint=ref_signal, dt=dt)

            u_hist.append(u_val)

            # Step Physics
            if self.current_system_id == "pendulum":
                # Pendulum B is (4,1), accepts u only.
                # In linear sim, disturbance injection requires augmented B or state injection.
                # For basic step response, we apply control u.
                u_vector = np.array([[u_val]])
            else:
                # Motor B is (2,2), accepts [u, dist]
                u_vector = np.array([[u_val], [dist_val]])

            y_real_vector = solver_real.step(u_vector)

            noise = np.random.normal(
                0, config.PRESET_SIM_PARAMS["noise_std"], size=y_real_vector.shape
            )
            y_meas = y_real_vector + noise

            if kf:
                kf.update(np.array([[u_val]]), y_meas)
                x_est_hist.append(kf.x_hat.flatten())

            y_real_hist.append(y_real_vector.flatten())

        return t_values, np.array(y_real_hist), np.array(x_est_hist), np.array(u_hist)

    def run_preset_dashboard(self):
        """Option 1: Time-Domain Response Dashboard (2x2)."""
        if not self.current_descriptor.supports_analysis:
            print("\nAnalysis & metrics are only available for linear systems.")
            input("Press Enter to return to menu...")
            return

        print(f"\nSimulating Time Response for {self.system_name}...")
        try:
            current_system = self.SystemClass(**self.active_params)
        except Exception as e:
            print(f"Error instantiating {self.system_name}: {e}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax1, ax2 = axes[0, 0], axes[0, 1]
        ax3, ax4 = axes[1, 0], axes[1, 1]

        if self.current_system_id == "dc_motor":
            labels = [
                "Speed (rad/s)",
                "Current (A)",
                "Voltage (V)",
                "Est. Disturbance (Nm)",
            ]
            indices = [0, 1]
            loop_controllers = self.controllers
        else:
            labels = [
                "Angle (rad)",
                "Cart Pos (m)",
                "Force (N)",
                "Ang. Velocity (rad/s)",
            ]
            indices = [2, 0]
            # LQR is unique, doesn't use the PID presets list
            loop_controllers = [{"name": "LQR Stabilizer", "color": "k"}]

        print("-" * 60)
        print("Simulating Controller Responses...")

        for ctrl in loop_controllers:
            t, y_real, x_est, u_hist = self.simulate_preset_system(current_system, ctrl)

            if len(y_real) > 0:
                ax1.plot(
                    t, y_real[:, indices[0]], label=ctrl["name"], color=ctrl["color"]
                )
                ax2.plot(
                    t, y_real[:, indices[1]], label=ctrl["name"], color=ctrl["color"]
                )
                ax3.plot(
                    t, u_hist, label=ctrl["name"], color=ctrl["color"], linestyle="--"
                )

                if self.current_system_id == "pendulum":
                    ax4.plot(t, y_real[:, 3], label=ctrl["name"], color=ctrl["color"])
                else:
                    if len(x_est) > 0:
                        ax4.plot(
                            t, x_est[:, -1], label=ctrl["name"], color=ctrl["color"]
                        )

        ax1.set_title(labels[0])
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        ax2.set_title(labels[1])
        ax2.grid(True, alpha=0.3)
        ax3.set_title(labels[2])
        ax3.grid(True, alpha=0.3)
        ax4.set_title(labels[3])
        ax4.grid(True, alpha=0.3)

        if self.current_system_id == "dc_motor" and self.dist_params["enabled"]:
            ax4.axhline(
                self.dist_params["magnitude"],
                color="k",
                linestyle=":",
                label="True Load",
            )
            ax4.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

    def run_analysis_dashboard(self):
        """Option 2: System Analysis (Freq, Poles, Phase)."""
        print(f"\nGenerating Analysis for {self.system_name}...")

        try:
            current_system = self.SystemClass(**self.active_params)
            ss = current_system.get_state_space()
        except Exception:
            print("System does not support state-space analysis.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax_bode, ax_pz = axes[0, 0], axes[0, 1]
        ax_kalman, ax_phase = axes[1, 0], axes[1, 1]

        # 1. Bode Plot (Plant V -> Output)
        w = np.logspace(*config.PLOT_PARAMS["bode_range"])
        # Input 0 -> Output 0 (Speed or Angle)
        out_idx = 2 if self.current_system_id == "pendulum" else 0
        mags, phases = ss.get_frequency_response(w, input_idx=0, output_idx=out_idx)
        ax_bode.semilogx(w, mags, "k-", lw=2)
        ax_bode.set_title("Bode Magnitude (Input -> Primary State)")
        ax_bode.set_xlabel("Frequency (rad/s)")
        ax_bode.set_ylabel("Magnitude (dB)")
        ax_bode.grid(True, which="both", alpha=0.3)

        # 2. Pole-Zero Map
        eigenvalues = np.linalg.eigvals(ss.A)
        ax_pz.scatter(
            eigenvalues.real,
            eigenvalues.imag,
            marker="x",
            color="r",
            s=100,
            label="Poles",
        )
        ax_pz.axhline(0, color="k", lw=1)
        ax_pz.axvline(0, color="k", lw=1)
        ax_pz.set_title("Pole-Zero Map (S-Plane)")
        ax_pz.set_xlabel("Real")
        ax_pz.set_ylabel("Imaginary")
        ax_pz.grid(True, alpha=0.3)
        ax_pz.legend()

        # 3. Kalman Estimate Error
        # Run a quick sim to get error data
        t, y_real, x_est, _ = self.simulate_preset_system(
            current_system, self.controllers[1]
        )  # Use PI/Balanced

        if len(x_est) > 0:
            err = y_real[:, out_idx] - x_est[:, out_idx]  # True - Est
            ax_kalman.plot(t, err, "b")
            ax_kalman.set_title("Kalman Estimation Error")
            ax_kalman.set_ylabel("Error")
            ax_kalman.grid(True, alpha=0.3)
        else:
            ax_kalman.text(0.5, 0.5, "No Estimator Data", ha="center")

        # 4. Phase Plane Trajectory
        if len(y_real) > 0:
            if self.current_system_id == "pendulum":
                # Theta (2) vs Theta_dot (3)
                ax_phase.plot(y_real[:, 2], y_real[:, 3], "g")
                ax_phase.set_xlabel("Angle (rad)")
                ax_phase.set_ylabel("Ang. Velocity (rad/s)")
            else:
                # Speed (0) vs Current (1)
                ax_phase.plot(y_real[:, 0], y_real[:, 1], "purple")
                ax_phase.set_xlabel("Speed (rad/s)")
                ax_phase.set_ylabel("Current (A)")

        ax_phase.set_title("Phase Plane Trajectory")
        ax_phase.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_custom_simulation(self):
        """Runs the adaptive step-size solver on user-defined non-linear equations."""
        print("\n--- Custom Non-Linear Simulation (Adaptive RK45) ---")
        eqn = input("Enter dx/dt = f(t, x, u): ").strip()
        try:
            dyn_func = make_system_func(eqn)
            x0 = np.zeros(config.CUSTOM_SIM_PARAMS["initial_state"]).flatten()
            solver = NonlinearSolver(dynamics_func=dyn_func, dt_min=1e-5, dt_max=0.1)
            step_time = config.CUSTOM_SIM_PARAMS["step_time"]

            def input_signal(t):
                return (
                    config.CUSTOM_SIM_PARAMS["step_magnitude"] if t > step_time else 0.0
                )

            print("Simulating...")
            t_vals, states = solver.solve_adaptive(
                t_end=config.CUSTOM_SIM_PARAMS["t_end"], x0=x0, u_func=input_signal
            )
            y_vals = states[:, 0] if states.ndim > 1 else states

            plt.figure(figsize=config.PLOT_PARAMS["figsize"])
            plt.plot(t_vals, y_vals, label=f"dx/dt = {eqn}")
            plt.title(f"Adaptive RK45 Simulation: {eqn}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()

        except Exception as e:
            print(f"\nError: {e}")

    def run_parameter_estimation(self):
        """
        Runs the Extended Kalman Filter (EKF) demo.
        Adapts based on the active system (Motor or Pendulum).
        """

        if not self.current_descriptor.supports_estimation:
            print("\nEstimation is not supported for this system.")
            input("Press Enter to return to menu...")
            return

        print(f"\n--- Parameter Estimation Demo ({self.system_name}) ---")

        # 1. Setup Configuration & Indices based on System
        if self.current_system_id == "dc_motor":
            est_cfg = config.ESTIMATION_PARAMS
            param_names = ["Inertia (J)", "Friction (b)"]
            param_keys = ["J", "b"]

            # States: [Speed, Current, J, b]
            # Measurement: [Speed, Current]
            def h_meas(x):
                return x[:2]

            x0_est = [
                0,
                0,
                np.log(est_cfg["initial_guess_J"]),
                np.log(est_cfg["initial_guess_b"]),
            ]
            plot_labels = ["Speed (rad/s)", "Current (A)"]
            true_indices = [0, 1]  # Index in True State vector
            est_indices = [0, 1]  # Index in Est State vector

        elif self.current_system_id == "pendulum":
            est_cfg = config.PENDULUM_ESTIMATION_PARAMS
            param_names = ["Mass (m)", "Length (l)"]
            param_keys = ["m", "l"]

            # States: [x, v, theta, omega, m, l]
            # Measurement: [x, theta] (We measure positions)
            def h_meas(x):
                return np.array([x[0], x[2]])

            x0_est = [
                0,
                0,
                0,
                0,
                np.log(est_cfg["initial_guess_m"]),
                np.log(est_cfg["initial_guess_l"]),
            ]
            plot_labels = ["Position (m)", "Angle (rad)"]
            true_indices = [0, 2]  # x is 0, theta is 2 in standard state
            est_indices = [0, 2]  # x is 0, theta is 2 in est state

        # 2. Setup Simulation
        dt = est_cfg["dt"]
        t_end = est_cfg["t_end"]
        true_params = est_cfg["true_system_params"]

        # Instantiate "True" System
        true_system = self.SystemClass(**true_params)

        # Get Dynamics for EKF
        f_dyn_est = true_system.get_parameter_estimation_func()

        # Initialize EKF
        Q = np.diag(est_cfg["Q_init"])
        R = np.diag(est_cfg["R"])

        ekf = ExtendedKalmanFilter(
            f_dyn_est, h_meas, Q, R, x0_est, p_init_scale=est_cfg["p_init_scale"]
        )

        # 3. Run Loop
        t_vals = np.linspace(0, t_end, int(t_end / dt))

        history = {
            "t": t_vals,
            "y1_true": [],
            "y1_est": [],
            "y2_true": [],
            "y2_est": [],
            "p1_est": [],
            "p2_est": [],
        }

        print("Simulating...")

        amp = est_cfg["input_amplitude"]
        period = est_cfg["input_period"]
        noise_std = est_cfg["sensor_noise_std"]

        # Initial State for Truth
        if self.current_system_id == "pendulum":
            x_true = np.zeros(4)  # [x, v, theta, omega]
        else:
            ss_true = true_system.get_state_space()
            solver_true = ExactSolver(ss_true.A, ss_true.B, ss_true.C, ss_true.D, dt=dt)

        for t in t_vals:
            # Input Signal
            if (t % period) < (period / 2.0):
                u_val = amp
            else:
                u_val = 0.0

            if self.current_system_id == "pendulum":
                u_vec = np.array([u_val])  # Force
            else:
                u_vec = np.array([[u_val], [0]])  # Voltage, Disturbance

            # Step "True" Reality
            if self.current_system_id == "pendulum":
                x_true = rk4_fixed_step(
                    pendulum_dynamics, x_true, u_val, dt, true_params
                )
                y_true_full = x_true
            else:
                y_true_full = solver_true.step(u_vec)

            # Generate Noisy Measurements
            # Map full true state to measurement space
            if self.current_system_id == "pendulum":
                # Measure x(0) and theta(2)
                meas_clean = np.array([y_true_full[0], y_true_full[2]])
            else:
                # Measure Speed(0) and Current(1)
                meas_clean = y_true_full.flatten()

            y_meas = meas_clean.reshape(-1, 1) + np.random.normal(0, noise_std, (2, 1))

            # EKF Step
            ekf.predict(np.array([[u_val]]), dt)
            x_hat = ekf.update(y_meas)

            # Store Data
            history["y1_true"].append(y_true_full[true_indices[0]])
            history["y1_est"].append(x_hat[est_indices[0]])
            history["y2_true"].append(y_true_full[true_indices[1]])
            history["y2_est"].append(x_hat[est_indices[1]])

            # Parameters are always the last 2 states in log form
            history["p1_est"].append(np.exp(x_hat[-2]))
            history["p2_est"].append(np.exp(x_hat[-1]))

        # 4. Plotting
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # State 1
        axes[0, 0].plot(t_vals, history["y1_true"], "k-", label="True")
        axes[0, 0].plot(t_vals, history["y1_est"], "r--", label="Est")
        axes[0, 0].set_title(f"State Tracking: {plot_labels[0]}")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # State 2
        axes[0, 1].plot(t_vals, history["y2_true"], "k-", label="True")
        axes[0, 1].plot(t_vals, history["y2_est"], "m--", label="Est")
        axes[0, 1].set_title(f"State Tracking: {plot_labels[1]}")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Param 1
        true_p1 = true_params[param_keys[0]]
        axes[1, 0].plot(t_vals, history["p1_est"], "b-", label="Estimate")
        axes[1, 0].axhline(true_p1, color="k", linestyle=":", label=f"True ({true_p1})")
        axes[1, 0].set_title(f"Estimation: {param_names[0]}")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Param 2
        true_p2 = true_params[param_keys[1]]
        axes[1, 1].plot(t_vals, history["p2_est"], "g-", label="Estimate")
        axes[1, 1].axhline(true_p2, color="k", linestyle=":", label=f"True ({true_p2})")
        axes[1, 1].set_title(f"Estimation: {param_names[1]}")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def run_ukf_demo(self):
        """Runs the Unscented Kalman Filter demo."""

        # Check for support
        if not hasattr(self.system, "get_nonlinear_dynamics"):
            print(f"\nUKF is not supported/implemented for {self.system_name}.")
            input("Press Enter to return to menu...")
            return

        if self.system:
            self.system.params = self.active_params.copy()

        print(f"\n--- Non-Linear State Estimation (UKF) - {self.system_name} ---")

        # 1. Load System-Specific Config
        if self.current_system_id == "pendulum":
            cfg = config.UKF_PENDULUM_PARAMS
            labels = ["Angle (rad)", "Velocity (rad/s)"]
        elif self.current_system_id == "dc_motor":
            cfg = config.UKF_MOTOR_PARAMS
            labels = ["Speed (rad/s)", "Current (A)"]
        else:
            print("Unknown system config.")
            return

        # 2. Setup UKF
        dt = cfg["dt"]

        # Get system-specific physics functions
        f_dyn, h_meas = self.system.get_nonlinear_dynamics()

        x0 = cfg["x0"]
        P0 = np.eye(len(x0)) * cfg["P0"]
        Q = np.diag(cfg["Q_diag"])
        R = np.diag(cfg["R_diag"])

        ukf = UnscentedKalmanFilter(
            f_dyn,
            h_meas,
            Q,
            R,
            x0,
            P0,
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            kappa=cfg["kappa"],
        )

        # 3. Simulation Loop
        t_vals = np.arange(0, cfg["t_end"], dt)
        true_states = []
        est_states = []
        measurements = []

        curr_x = np.array(x0)

        print("Simulating...")
        for t in t_vals:
            # Input Signal
            if self.current_system_id == "dc_motor":
                # Sine wave to test stiction (cross zero speed)
                u = 2.0 * np.sin(2.0 * t)
            else:
                u = 0.0  # Free swing for pendulum

            # Step True Physics (using the same non-linear function for "Truth" here)
            curr_x = f_dyn(curr_x, u, dt)
            true_states.append(curr_x)

            # Generate Noisy Measurement
            z_clean = h_meas(curr_x)
            z_noisy = z_clean + np.random.normal(
                0, cfg["noise_std"], size=z_clean.shape
            )
            measurements.append(z_noisy)

            # UKF Step
            ukf.predict(u, dt)
            est_x = ukf.update(z_noisy)
            est_states.append(est_x)

        true_states = np.array(true_states)
        est_states = np.array(est_states)
        measurements = np.array(measurements)

        # 4. Plotting
        plt.figure(figsize=(10, 8))

        # Subplot 1: Primary State (Angle or Speed)
        plt.subplot(2, 1, 1)
        plt.plot(t_vals, true_states[:, 0], "k-", label="True State")

        # Show measurements if they correspond to this state
        # Pendulum measures state 0 (Angle), Motor measures both
        plt.plot(t_vals, measurements[:, 0], "g.", alpha=0.3, label="Noisy Measure")

        plt.plot(t_vals, est_states[:, 0], "r--", linewidth=2, label="UKF Estimate")
        plt.ylabel(labels[0])
        plt.title(f"UKF Estimation: {labels[0]}")
        plt.legend()
        plt.grid(True)

        # Subplot 2: Secondary State
        plt.subplot(2, 1, 2)
        plt.plot(t_vals, true_states[:, 1], "k-", label="True State")
        plt.plot(t_vals, est_states[:, 1], "b--", linewidth=2, label="UKF Estimate")
        plt.ylabel(labels[1])
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def run_mpc_demo(self):
        """Runs the Model Predictive Control demo."""

        if not hasattr(self.system, "get_mpc_model"):
            print(f"\nMPC is not supported for {self.system_name}.")
            input("Press Enter to return to menu...")
            return

        if self.system:
            self.system.params = self.active_params.copy()

        print(f"\n--- Model Predictive Control (MPC) - {self.system_name} ---")

        # 1. Setup based on System Type
        if self.current_system_id == "dc_motor":
            cfg = config.MPC_MOTOR_PARAMS
            dt = cfg["dt"]

            # Linear Model -> ADMM Solver
            A_d, B_d = self.system.get_mpc_model(dt)
            model_func = None  # Not needed for ADMM

            x0 = np.array([0.0, 0.0])  # [Speed, Current]
            ref = np.array([cfg["target_speed"], 0.0])

            plot_labels = ["Speed (rad/s)", "Voltage (V)"]
            plot_idx = 0

        elif self.current_system_id == "pendulum":
            cfg = config.MPC_PENDULUM_PARAMS
            dt = cfg["dt"]

            # Nonlinear Model -> iLQR Solver
            model_func = self.system.get_mpc_model(dt)
            A_d, B_d = None, None  # Not needed for iLQR

            # Start hanging DOWN (Pi), Goal is UP (0)
            x0 = np.array([0.0, 0.0, cfg["start_theta"], 0.0])
            ref = np.array([0.0, 0.0, 0.0, 0.0])

            plot_labels = ["Angle (rad)", "Force (N)"]
            plot_idx = 2  # Angle is index 2

        else:
            return

        # 2. Initialize Controller
        Q = np.diag(cfg["Q_diag"])
        R = np.diag(cfg["R_diag"])

        mpc = ModelPredictiveControl(
            model_func=model_func,  # Used if Nonlinear
            A=A_d,
            B=B_d,  # Used if Linear
            x0=x0,
            horizon=cfg["horizon"],
            dt=dt,
            Q=Q,
            R=R,
            u_min=cfg["u_min"],
            u_max=cfg["u_max"],
        )

        # 3. Simulation Loop
        # We simulate for twice the horizon to see steady state
        t_vals = np.arange(0, dt * cfg["horizon"] * 3, dt)
        x_hist = []
        u_hist = []

        curr_x = x0.copy()

        print(f"Solving using {mpc.mode.upper()}...")
        if mpc.mode == "nonlinear":
            print("Note: iLQR Swing-up may take a moment to compute...")

        for t in t_vals:
            # Optimize
            u_opt = mpc.optimize(curr_x, ref, iterations=cfg["iterations"])

            # Store
            x_hist.append(curr_x)
            u_hist.append(u_opt[0])

            # Advance Physics (Use the same model as the controller for the demo)
            if mpc.mode == "linear":
                curr_x = A_d @ curr_x + B_d @ u_opt
            else:
                curr_x = model_func(curr_x, u_opt, dt)

        x_hist = np.array(x_hist)
        u_hist = np.array(u_hist)

        # 4. Plotting
        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t_vals, x_hist[:, plot_idx], "b-", linewidth=2, label="System Output")
        plt.axhline(ref[plot_idx], color="k", linestyle="--", label="Target")

        if self.current_system_id == "pendulum":
            # Draw the "Upright" line
            plt.axhline(0, color="g", linestyle=":", alpha=0.5)
            plt.axhline(np.pi, color="r", linestyle=":", alpha=0.5, label="Down")

        plt.title(f"MPC Response ({mpc.mode.upper()})")
        plt.ylabel(plot_labels[0])
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.step(t_vals, u_hist, "r-", where="post", label="Control Input")
        plt.axhline(cfg["u_max"], color="k", linestyle="--", label="Limits")
        plt.axhline(cfg["u_min"], color="k", linestyle="--")
        plt.ylabel(plot_labels[1])
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def run_interactive_lab(self):
        if not self.current_descriptor.supports_interactive_lab:
            print("\nInteractive Lab not supported for this system.")
            input("Press Enter to continue...")
            return

        params = self.active_params.copy()

        lab = InteractiveLab(
            system_descriptor=self.current_descriptor,
            params=params,
            dt=0.01,
        )

        lab.initialize()
        lab.init_visualization()

        if self.current_system_id == "dc_motor":
            omega_ref = params.get(
                "omega_ref", config.INTERACTIVE_LAB_PARAMS["omega_ref"]
            )
            target_config = config.CONTROLLERS[2]

            print(f"\n[INFO] Loaded Controller: {target_config['name']}")
            print(
                f"       Gains: Kp={target_config['Kp']}, Ki={target_config['Ki']}, Kd={target_config['Kd']}"
            )

            pid = PIDController(
                Kp=target_config["Kp"],
                Ki=target_config["Ki"],
                Kd=target_config["Kd"],
                derivative_on_measurement=True,
                output_limits=(None, None),
            )

            last_t = 0.0

            def controller_wrapper(state, t):
                nonlocal last_t
                dt = t - last_t
                if dt <= 0:
                    dt = config.INTERACTIVE_LAB_PARAMS["controller_min_dt"]
                last_t = t

                return pid.update(measurement=state[0], setpoint=omega_ref, dt=dt)

            lab.set_auto_controller(controller_wrapper)

        if self.current_system_id == "pendulum":
            # Instantiate the current system class dynamically
            current_sys_instance = self.SystemClass(**self.active_params)

            # Check if it supports LQR (look for the method)
            if hasattr(current_sys_instance, "dlqr_gain"):
                K = current_sys_instance.dlqr_gain()
                lab.set_auto_controller(pendulum_lqr_controller(K))
            else:
                print("[WARN] System does not support LQR. Auto-control disabled.")

            x0 = np.array(config.INTERACTIVE_LAB_PARAMS["ekf_x0"])
            Q = np.diag(config.INTERACTIVE_LAB_PARAMS["ekf_Q_diag"])
            R = np.diag(config.INTERACTIVE_LAB_PARAMS["ekf_R_diag"])

            # Check if it supports continuous dynamics for EKF
            if hasattr(current_sys_instance, "dynamics_continuous") and hasattr(
                current_sys_instance, "measurement"
            ):
                ekf = ExtendedKalmanFilter(
                    f_dynamics=current_sys_instance.dynamics_continuous,
                    h_measurement=current_sys_instance.measurement,
                    Q=Q,
                    R=R,
                    x0=x0,
                )
                lab.set_estimator(
                    ekf, measurement_func=current_sys_instance.measurement
                )

        print("\nInteractive Lab running (headless mode).")
        print(f"Control mode: {lab.control_mode}")
        print("Press Ctrl+C to stop.\n")

        try:
            if lab.control_mode == "MANUAL":
                print("\nControls:")
                print("  a/d : decrease/increase input")
                print("  s   : zero input")
                print("  m   : manual mode")
                print("  o   : auto mode")
                print("  q   : quit\n")

            else:
                print("\nAUTO mode enabled.")
                print("Press Ctrl+C to stop or wait for completion.\n")

            while lab.running and lab.status == "RUNNING":
                lab.step()
                lab.update_visualization()
                time.sleep(lab.dt)

                actual_u = lab.last_u

                if lab.descriptor.system_id == "dc_motor":
                    omega, current = lab.state
                    print(
                        f"\rω={omega:.2f}, i={current:.2f}, u={actual_u:.2f}",
                        end="",
                    )
                elif lab.descriptor.system_id == "pendulum":
                    theta = lab.state[2]
                    x_pos = lab.state[0]
                    print(
                        f"\rθ={theta:.3f}, x={x_pos:.3f}, u={actual_u:.2f}",
                        end="",
                    )

            plt.ioff()
            plt.show()

            print(f"\nLab finished with status: {lab.status}")
            if lab.failure_reason:
                print(f"Reason: {lab.failure_reason}")

        except KeyboardInterrupt:
            pass
        except EOFError:
            print("\n\nClosing the terminal...")
            time.sleep(0.1)
            kill()

        # print(f"\nLab finished with status: {lab.status}")
        # if lab.failure_reason:
        #     print(f"Reason: {lab.failure_reason}")

        input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    app = PyControlsApp()
    KILLED = False
    try:
        app.main_menu()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        time.sleep(0.5)
    except EOFError:
        print("\n\nClosing the terminal...")
        time.sleep(0.2)
        KILLED = True
    finally:
        flush()
        if KILLED:
            kill()
        else:
            app.clear_screen()
            stop()
