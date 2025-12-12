import importlib
import inspect
import os
import pkgutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Ensure local modules are found
sys.path.append(os.getcwd())

import config
from core.analysis import get_stability_margins, get_step_metrics
from core.ekf import ExtendedKalmanFilter
from core.estimator import KalmanFilter
from core.math_utils import make_system_func
from core.solver import ExactSolver, NonlinearSolver
from core.state_space import StateSpace


# --- Dynamic System Loader ---
def load_available_systems():
    """
    Scans the 'systems' package for modules and classes that implement
    the expected interface.
    """
    systems = {}
    systems_path = os.path.join(os.getcwd(), "systems")

    for _, name, _ in pkgutil.iter_modules([systems_path]):
        module_name = f"systems.{name}"
        try:
            module = importlib.import_module(module_name)
            for member_name, member_obj in inspect.getmembers(module, inspect.isclass):
                # Duck Typing: Check if it has the required methods
                # Note: We still check for old methods to ensure backward compatibility
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
    def __init__(self):
        self.available_systems = load_available_systems()
        if not self.available_systems:
            print("Error: No valid system classes found in systems/ folder.")
            sys.exit(1)

        self.system_name = (
            "DCMotor"
            if "DCMotor" in self.available_systems
            else next(iter(self.available_systems))
        )
        self.SystemClass = self.available_systems[self.system_name]

        temp_instance = self.SystemClass()
        self.active_params = temp_instance.params.copy()

        if self.system_name == "DCMotor" and hasattr(config, "MOTOR_PARAMS"):
            self.active_params.update(config.MOTOR_PARAMS)

        self.controllers = config.CONTROLLERS.copy()
        self.sim_params = config.SIM_PARAMS.copy()
        self.dist_params = config.DISTURBANCE_PARAMS.copy()
        self.running = True

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self):
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
        self.clear_screen()
        while self.running:
            self.print_header()
            print("[1] Run Standard Simulation (MIMO + Kalman Filter)")
            print("[2] Run Custom Non-Linear Simulation (Adaptive RK45)")
            print("[3] Edit System Parameters")
            print("[4] Edit Disturbance Settings")
            print("[5] Switch Active System")
            print("[6] Run Parameter Estimation Demo (EKF)")
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
            elif choice == "q":
                self.running = False
            else:
                input("Invalid option. Press Enter...")

    def switch_system_menu(self):
        self.clear_screen()
        print("\nAvailable Systems:")
        names = list(self.available_systems.keys())
        for i, name in enumerate(names):
            print(f"[{i + 1}] {name}")
        print("[b] Back")
        sel = input("\nSelect System ID: ").strip()
        if sel == "b":
            return
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(names):
                new_name = names[idx]
                if new_name != self.system_name:
                    self.system_name = new_name
                    self.SystemClass = self.available_systems[new_name]
                    self.active_params = self.SystemClass().params.copy()
                    if new_name == "DCMotor" and hasattr(config, "MOTOR_PARAMS"):
                        self.active_params.update(config.MOTOR_PARAMS)
                    print(f"Switched to {new_name}")
                    time.sleep(1)
                    self.clear_screen()
        except ValueError:
            pass

    def edit_params_menu(self):
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
        dt = self.sim_params["dt"]
        t_end = self.sim_params["t_end"]

        # 1. Setup TRUE System (2-State MIMO: Speed, Current)
        if hasattr(system_instance, "get_state_space"):
            ss_real = system_instance.get_state_space()
            solver_real = ExactSolver(ss_real.A, ss_real.B, ss_real.C, ss_real.D, dt)
        else:
            # Fallback for old systems (returns empty arrays)
            return np.array([]), np.array([]), np.array([])

        # 2. Setup KALMAN FILTER (3-State Augmented: Speed, Current, Disturbance)
        if hasattr(system_instance, "get_augmented_state_space"):
            ss_aug = system_instance.get_augmented_state_space()
            # Use solver logic to discretize the Augmented Matrices
            solver_aug_math = ExactSolver(ss_aug.A, ss_aug.B, ss_aug.C, ss_aug.D, dt)

            # Tuning:
            Q = np.diag([1e-4, 1e-4, 1e-2])  # Process Noise
            R = np.diag([0.01, 0.01])  # Sensor Noise

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
            # A. Disturbance Logic
            dist_torque = 0.0
            if self.dist_params["enabled"] and t >= self.dist_params["time"]:
                dist_torque = self.dist_params["magnitude"]

            # B. Controller Step
            ref_speed = self.sim_params["step_volts"] if t > 0 else 0

            # Feedback: Use ESTIMATE if available, else REAL (cheating)
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

            # C. Physics Step (Real World)
            # Input: [Voltage, Disturbance]
            u_real = np.array([[voltage], [dist_torque]])
            y_real_vector = solver_real.step(u_real)  # [Speed, Current]

            # D. Sensor Noise
            noise = np.random.normal(0, 0.1, size=2)
            y_meas = y_real_vector + noise

            # E. Kalman Filter Step
            if kf:
                # Filter sees: [Voltage] only
                u_kf = np.array([[voltage]])
                x_est = kf.update(u_kf, y_meas)  # [Speed, Current, Dist_Est]
                x_est_hist.append(x_est)

            y_real_hist.append(y_real_vector)

        return t_values, np.array(y_real_hist), np.array(x_est_hist)

    def run_preset_dashboard(self):
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

        # Plot Styling
        if self.dist_params["enabled"]:
            t_dist = self.dist_params["time"]
            # Get style from config
            style = config.PLOT_PARAMS["marker_style"]

            # Add vertical line to Speed, Current and Kalman plots
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

        # Bode Plot (Direct Matrix Method)
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
        print("\n--- Parameter Estimation Demo (EKF) ---")
        print("Goal: Estimate Inertia (J) and Friction (b) from scratch.")

        # Load parameters from config
        est_cfg = config.ESTIMATION_PARAMS
        dt = est_cfg["dt"]
        t_end = est_cfg["t_end"]
        true_params = est_cfg["true_system_params"]

        # 1. Setup the TRUE System (The Reality)
        # Use parameters defined in config
        true_motor = self.SystemClass(**true_params)
        ss_true = true_motor.get_state_space()
        solver_true = ExactSolver(ss_true.A, ss_true.B, ss_true.C, ss_true.D, dt=dt)

        # 2. Setup the EKF (The Learner)
        # Function that maps [w, i, J, b] -> [dot_w, dot_i, 0, 0]
        f_dyn = true_motor.get_parameter_estimation_func()

        # Measurement function: We measure Speed and Current [w, i]
        # h(x) = [x[0], x[1]]
        def h_meas(x):
            return x[:2]

        # Initial Guess (configured in config.py)
        # The EKF uses Log(J) and Log(b) in the state vector, so we apply log here.
        x0_est = [
            0,
            0,
            np.log(est_cfg["initial_guess_J"]),
            np.log(est_cfg["initial_guess_b"]),
        ]

        # Tuning
        Q = np.diag(est_cfg["Q_init"])  # Process noise
        R = np.diag(est_cfg["R"])  # Sensor noise

        ekf = ExtendedKalmanFilter(
            f_dyn, h_meas, Q, R, x0_est, p_init_scale=est_cfg["p_init_scale"]
        )

        # 3. Simulation Loop
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
            # Input Square Wave
            if (t % period) < (period / 2.0):
                volts = amp
            else:
                volts = 0.0
            u = np.array([[volts], [0]])  # True system has Load input too (0)

            # Adaptive Q Logic
            if est_cfg["adaptive_enabled"]:
                phi = (1 + np.sqrt(5)) / 2
                if t < t_end / phi:
                    # Search Mode
                    ekf.Q = np.diag(est_cfg["Q_search"])
                else:
                    # Lock-in Mode
                    ekf.Q = np.diag(est_cfg["Q_lock"])

            # A. Real World Step
            y_true = solver_true.step(u)  # [Speed, Current]
            y_meas = np.array(y_true).reshape(-1, 1) + np.random.normal(
                0, noise_std, (2, 1)
            )

            # B. EKF Step
            # Predict
            ekf.predict(np.array([[volts]]), dt)
            # Correct
            x_hat = ekf.update(y_meas)

            # Store Data
            speed_true.append(y_true[0])
            speed_est.append(x_hat[0])
            current_true.append(y_true[1])
            current_est.append(x_hat[1])
            J_est_hist.append(np.exp(x_hat[2]))
            b_est_hist.append(np.exp(x_hat[3]))

        # 4. Plotting
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Speed Convergence
        axes[0, 0].plot(t_vals, speed_true, "k-", label="True Speed")
        axes[0, 0].plot(t_vals, speed_est, "r--", label="EKF Est")
        axes[0, 0].set_title("State Tracking: Speed")
        axes[0, 0].legend()

        # Current Convergence
        axes[0, 1].plot(t_vals, current_true, "k-", label="True Current")
        axes[0, 1].plot(t_vals, current_est, "m--", label="EKF Current")
        axes[0, 1].set_title("State Tracking: Current")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Inertia Estimation
        true_J = true_params["J"]
        axes[1, 0].plot(t_vals, J_est_hist, "b-", label="Est J")
        axes[1, 0].axhline(true_J, color="k", linestyle=":", label=f"True J ({true_J})")
        axes[1, 0].set_title("Parameter Estimation: Inertia (J)")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Friction Estimation
        true_b = true_params["b"]
        axes[1, 1].plot(t_vals, b_est_hist, "g-", label="Est b")
        axes[1, 1].axhline(true_b, color="k", linestyle=":", label=f"True b ({true_b})")
        axes[1, 1].set_title("Parameter Estimation: Friction (b)")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = PyControlsApp()
    try:
        app.main_menu()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        time.sleep(2)
    except EOFError:
        print("\n\nClosing the terminal...")
        time.sleep(1)
    finally:
        app.clear_screen()
