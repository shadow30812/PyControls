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
from core.ekf import ExtendedKalmanFilter
from core.estimator import KalmanFilter
from core.math_utils import make_system_func
from core.mpc import ModelPredictiveControl
from core.solver import ExactSolver, NonlinearSolver
from core.ukf import UnscentedKalmanFilter
from exit import flush, kill, stop
from modules.interactive_lab import InteractiveLab, simple_dc_motor_pid
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
            print("[1] Run Standard Simulation (MIMO + Kalman Filter)")
            print("[2] Run Custom Non-Linear Simulation (Adaptive RK45)")
            print("[3] Edit System Parameters")
            print("[4] Edit Disturbance Settings")
            print("[5] Switch Active System")
            print("[6] Run Parameter Estimation Demo (EKF)")
            print("[7] Run Nonlinear State Est. Demo (UKF)")
            print("[8] Run Model Predictive Control Demo (MPC)")
            print("[9] Interactive Lab")
            print("[q] Exit")

            choice = input("\nSelect Option: ").strip()

            if choice == "1":
                self.run_preset_dashboard()
            elif choice == "2":
                self.run_custom_simulation()
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
                    else:
                        self.system = None

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
        """
        Runs the simulation for a specific controller configuration.

        Args:
            system_instance: The system object to simulate.
            ctrl_config: Dictionary containing PID gains.

        Returns:
            tuple: (time_array, real_output_history, estimate_history)
        """

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

        if hasattr(system_instance, "get_augmented_state_space"):
            ss_aug = system_instance.get_augmented_state_space()
            solver_aug_math = ExactSolver(ss_aug.A, ss_aug.B, ss_aug.C, ss_aug.D, dt)

            Q = np.diag([1e-4, 1e-4, 1e-2])
            R = np.diag([0.01, 0.01])

            kf = KalmanFilter(
                solver_aug_math.Phi, solver_aug_math.Gamma, ss_aug.C, Q, R, x0=[0, 0, 0]
            )
        else:
            kf = None

        t_values = np.linspace(0, t_end, int(t_end / dt))
        y_real_hist = []
        x_est_hist = []

        integral_error = 0.0
        prev_error = 0.0

        for t in t_values:
            dist_torque = 0.0
            if self.dist_params["enabled"] and t >= self.dist_params["time"]:
                dist_torque = self.dist_params["magnitude"]

            ref_speed = self.sim_params["step_volts"] if t > 0 else 0

            if kf:
                speed_feedback = kf.x_hat[0, 0]
            else:
                speed_feedback = solver_real.x[0, 0]

            error = ref_speed - speed_feedback
            integral_error += error * dt
            derivative = (error - prev_error) / dt
            prev_error = error

            voltage = (
                (ctrl_config["Kp"] * error)
                + (ctrl_config["Ki"] * integral_error)
                + (ctrl_config["Kd"] * derivative)
            )

            voltage = np.clip(voltage, -12, 12)

            u_real = np.array([[voltage], [dist_torque]])
            y_real_vector = solver_real.step(u_real)

            noise = np.random.normal(0, 0.1, size=2)
            y_meas = y_real_vector + noise

            if kf:
                u_kf = np.array([[voltage]])
                x_est = kf.update(u_kf, y_meas)
                x_est_hist.append(x_est)

            y_real_hist.append(y_real_vector)

        return t_values, np.array(y_real_hist), np.array(x_est_hist)

    def run_preset_dashboard(self):
        """Executes the standard simulation with multiple controllers and plots results."""

        if not self.current_descriptor.supports_analysis:
            print("\nAnalysis & metrics are only available for linear systems.")
            input("Press Enter to return to menu...")
            return

        print(f"\nInitializing MIMO Simulation for {self.system_name}...")

        try:
            current_system = self.SystemClass(**self.active_params)
        except Exception as e:
            print(f"Error instantiating {self.system_name}: {e}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax_speed = axes[0, 0]
        ax_current = axes[0, 1]
        ax_dist = axes[1, 0]
        ax_bode = axes[1, 1]

        print("-" * 60)
        print("Simulating Controller Responses...")

        for ctrl in self.controllers:
            t, y_real, x_est = self.simulate_preset_system(current_system, ctrl)

            if len(y_real) > 0:
                ax_speed.plot(
                    t, y_real[:, 0], label=f"{ctrl['name']} (Real)", color=ctrl["color"]
                )
                ax_current.plot(
                    t, y_real[:, 1], label=f"{ctrl['name']}", color=ctrl["color"]
                )

                if len(x_est) > 0:
                    ax_dist.plot(
                        t,
                        x_est[:, 2],
                        label=f"{ctrl['name']} (Est Load)",
                        color=ctrl["color"],
                        alpha=0.6,
                    )

        if self.dist_params["enabled"]:
            t_dist = self.dist_params["time"]
            style = config.PLOT_PARAMS["marker_style"]

            ax_speed.axvline(x=t_dist, **style)
            ax_current.axvline(x=t_dist, **style)
            ax_dist.axvline(x=t_dist, **style)

            ax_speed.text(
                t_dist + 0.1,
                ax_speed.get_ylim()[0] * 0.9,
                "Load Applied",
                fontsize=8,
                rotation=90,
                alpha=0.6,
            )

        ax_speed.set_title("Speed Response (w/ Noise)")
        ax_speed.set_ylabel("Speed (rad/s)")
        ax_speed.grid(True, alpha=0.3)
        ax_speed.legend(fontsize=8)

        ax_current.set_title("Current Response")
        ax_current.set_ylabel("Current (A)")
        ax_current.grid(True, alpha=0.3)
        ax_current.legend(fontsize=8)

        ax_dist.set_title("Kalman Disturbance Estimation")
        ax_dist.set_ylabel("Est. Torque (Nm)")
        ax_dist.grid(True, alpha=0.3)
        if self.dist_params["enabled"]:
            ax_dist.axhline(
                y=self.dist_params["magnitude"],
                color="k",
                linestyle=":",
                label="True Load",
            )
            ax_dist.legend(fontsize=8)

        if hasattr(current_system, "get_state_space"):
            ss = current_system.get_state_space()
            w = np.logspace(-1, 3, 500)
            mags, phases = ss.get_frequency_response(w, input_idx=0, output_idx=0)
            ax_bode.semilogx(w, mags, color="k", label="Plant V->w")
            ax_bode.set_title("Plant Frequency Response (V -> Speed)")
            ax_bode.set_xlabel("Frequency (rad/s)")
            ax_bode.set_ylabel("Magnitude (dB)")
            ax_bode.grid(True, alpha=0.3)

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
        Estimates the system's inertia (J) and friction (b) in real-time.
        """

        if not self.current_descriptor.supports_estimation:
            print("\nEstimation is not supported for this system.")
            input("Press Enter to return to menu...")
            return

        print("\n--- Parameter Estimation Demo (EKF) ---")
        print("Goal: Estimate Inertia (J) and Friction (b) from scratch.")

        est_cfg = config.ESTIMATION_PARAMS
        dt = est_cfg["dt"]
        t_end = est_cfg["t_end"]
        true_params = est_cfg["true_system_params"]

        true_motor = self.SystemClass(**true_params)
        ss_true = true_motor.get_state_space()
        solver_true = ExactSolver(ss_true.A, ss_true.B, ss_true.C, ss_true.D, dt=dt)

        f_dyn = true_motor.get_parameter_estimation_func()

        def h_meas(x):
            return x[:2]

        x0_est = [
            0,
            0,
            np.log(est_cfg["initial_guess_J"]),
            np.log(est_cfg["initial_guess_b"]),
        ]

        Q = np.diag(est_cfg["Q_init"])
        R = np.diag(est_cfg["R"])

        ekf = ExtendedKalmanFilter(
            f_dyn, h_meas, Q, R, x0_est, p_init_scale=est_cfg["p_init_scale"]
        )

        t_vals = np.linspace(0, t_end, int(t_end / dt))

        J_est_hist = []
        b_est_hist = []
        speed_true = []
        speed_est = []
        current_true = []
        current_est = []

        print("Simulating... (This uses Complex Step Differentiation!)")

        amp = est_cfg["input_amplitude"]
        period = est_cfg["input_period"]
        noise_std = est_cfg["sensor_noise_std"]

        for t in t_vals:
            if (t % period) < (period / 2.0):
                volts = amp
            else:
                volts = 0.0
            u = np.array([[volts], [0]])

            if est_cfg["adaptive_enabled"]:
                phi = (1 + np.sqrt(5)) / 2
                if t < t_end / phi:
                    ekf.Q = np.diag(est_cfg["Q_search"])
                else:
                    ekf.Q = np.diag(est_cfg["Q_lock"])

            y_true = solver_true.step(u)
            y_meas = np.array(y_true).reshape(-1, 1) + np.random.normal(
                0, noise_std, (2, 1)
            )

            ekf.predict(np.array([[volts]]), dt)
            x_hat = ekf.update(y_meas)

            speed_true.append(y_true[0])
            speed_est.append(x_hat[0])
            current_true.append(y_true[1])
            current_est.append(x_hat[1])
            J_est_hist.append(np.exp(x_hat[2]))
            b_est_hist.append(np.exp(x_hat[3]))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(t_vals, speed_true, "k-", label="True Speed")
        axes[0, 0].plot(t_vals, speed_est, "r--", label="EKF Est")
        axes[0, 0].set_title("State Tracking: Speed")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(t_vals, current_true, "k-", label="True Current")
        axes[0, 1].plot(t_vals, current_est, "m--", label="EKF Current")
        axes[0, 1].set_title("State Tracking: Current")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        true_J = true_params["J"]
        axes[1, 0].plot(t_vals, J_est_hist, "b-", label="Est J")
        axes[1, 0].axhline(true_J, color="k", linestyle=":", label=f"True J ({true_J})")
        axes[1, 0].set_title("Parameter Estimation: Inertia (J)")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        true_b = true_params["b"]
        axes[1, 1].plot(t_vals, b_est_hist, "g-", label="Est b")
        axes[1, 1].axhline(true_b, color="k", linestyle=":", label=f"True b ({true_b})")
        axes[1, 1].set_title("Parameter Estimation: Friction (b)")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def run_ukf_demo(self):
        """Runs the Unscented Kalman Filter demo."""

        if not self.current_descriptor.supports_estimation:
            print("\nEstimation is not supported for this system.")
            input("Press Enter to return to menu...")
            return

        print("\n--- Non-Linear State Estimation (UKF) ---")

        cfg = config.UKF_PARAMS
        dt = cfg["dt"]

        def pendulum_dynamics(x, u, dt):
            g = 9.81
            L = 1.0
            theta, omega = x
            theta_next = theta + omega * dt
            omega_next = omega - (g / L) * np.sin(theta) * dt
            return np.array([theta_next, omega_next])

        def measure(x):
            return np.array([x[0]])

        x0 = [np.pi / 2, 0]
        P0 = np.eye(2) * 0.1
        Q = np.diag(cfg["Q"])
        R = np.diag(cfg["R"])

        ukf = UnscentedKalmanFilter(
            pendulum_dynamics,
            measure,
            Q,
            R,
            x0,
            P0,
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            kappa=cfg["kappa"],
        )

        t_vals = np.arange(0, cfg["t_end"], dt)
        true_states = []
        est_states = []

        curr_x = np.array(x0)

        print("Simulating Pendulum...")
        for t in t_vals:
            curr_x = pendulum_dynamics(curr_x, 0, dt)
            true_states.append(curr_x)

            z = np.array([curr_x[0]]) + np.random.normal(0, cfg["noise_std"])

            ukf.predict(0, dt)
            est_x = ukf.update(z)
            est_states.append(est_x)

        true_states = np.array(true_states)
        est_states = np.array(est_states)

        plt.figure(figsize=(10, 6))
        plt.plot(t_vals, true_states[:, 0], "k-", label="True Angle")
        plt.plot(t_vals, est_states[:, 0], "r--", label="UKF Angle")
        plt.plot(t_vals, true_states[:, 1], "g-", alpha=0.5, label="True Velocity")
        plt.plot(t_vals, est_states[:, 1], "b--", alpha=0.5, label="UKF Velocity")
        plt.title("UKF Nonlinear Estimation (Pendulum)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def run_mpc_demo(self):
        """Runs the Model Predictive Control demo."""

        if not self.current_descriptor.supports_mpc:
            print("\nMPC is not available for this system.")
            input("Press Enter to return to menu...")
            return

        print("\n--- Model Predictive Control (MPC) ---")

        cfg = config.MPC_PARAMS
        dt = cfg["dt"]

        A_lin = np.array([[0.9]])
        B_lin = np.array([[dt]])

        def simple_model(x, u, dt):
            return 0.9 * x + u * dt

        x0 = np.array([0.0])

        Q = np.diag([cfg["Q_weight"][0]])
        R = np.diag(cfg["R_weight"])

        mpc = ModelPredictiveControl(
            model_func=simple_model,
            x0=x0,
            horizon=cfg["horizon"],
            dt=dt,
            Q=Q,
            R=R,
            u_min=cfg["u_min"],
            u_max=cfg["u_max"],
            A=A_lin,
            B=B_lin,
        )

        ref = np.array([10.0])

        t_vals = np.arange(0, cfg["t_end"], dt)
        x_hist = []
        u_hist = []

        curr_x = x0

        print(f"Optimizing Control Trajectories using {mpc.mode.upper()} solver...")

        for t in t_vals:
            u_opt = mpc.optimize(curr_x, ref, iterations=cfg["iterations"])

            curr_x = simple_model(curr_x, u_opt, dt)

            x_hist.append(curr_x[0])
            u_hist.append(u_opt[0])

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t_vals, x_hist, "b-", label="System Output")
        plt.axhline(ref[0], color="k", linestyle="--", label="Setpoint")
        plt.title(f"MPC Tracking Performance ({mpc.mode.upper()} Solver)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.step(t_vals, u_hist, "r-", label="Control Input (u)")
        plt.axhline(cfg["u_max"], color="k", linestyle=":", label="Limits")
        plt.axhline(cfg["u_min"], color="k", linestyle=":")
        plt.title("Control Effort (Constrained)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_interactive_lab(self):
        if not self.current_descriptor.supports_interactive_lab:
            print("\nInteractive Lab not supported for this system.")
            input("Press Enter to continue...")
            return

        if self.current_system_id == "dc_motor":
            params = config.MOTOR_PARAMS
        elif self.current_system_id == "pendulum":
            params = config.PENDULUM_PARAMS
        else:
            params = {}

        lab = InteractiveLab(
            system_descriptor=self.current_descriptor,
            params=params,
            dt=0.01,
        )

        lab.initialize()
        lab.init_visualization()

        if self.current_system_id == "dc_motor":
            controller = simple_dc_motor_pid(omega_ref=1.0, Kp=2.0)
            lab.set_auto_controller(controller)

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

            while lab.status == "RUNNING":
                lab.step()
                lab.update_visualization()
                time.sleep(lab.dt)
                if lab.descriptor.system_id == "dc_motor":
                    omega, current = lab.state
                    print(
                        f"\rω={omega:.2f}, i={current:.2f}, u={lab.manual_input:.2f}",
                        end="",
                    )
                elif lab.descriptor.system_id == "pendulum":
                    theta, theta_dot = lab.state
                    print(
                        f"\rθ={theta:.3f}, θ̇={theta_dot:.3f}, u={lab.manual_input:.2f}",
                        end="",
                    )

            plt.ioff()
            plt.show()

            print(f"Lab finished with status: {lab.status}")
            if lab.failure_reason:
                print(f"Reason: {lab.failure_reason}")

        except KeyboardInterrupt:
            pass
        except EOFError:
            print("\n\nClosing the terminal...")
            time.sleep(0.1)
            kill()

        print(f"\nLab finished with status: {lab.status}")
        if lab.failure_reason:
            print(f"Reason: {lab.failure_reason}")

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
