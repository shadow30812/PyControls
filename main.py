import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
import helpers.config as config
from core.analysis import get_stability_margins
from core.control_utils import PIDController
from core.ekf import ExtendedKalmanFilter
from core.estimator import KalmanFilter
from helpers.config import BATTERY_KF, BATTERY_PID
from helpers.exit import flush, kill, stop
from helpers.plot import (
    plot_analysis_dashboard,
    plot_custom_simulation,
    plot_estimation_history,
    plot_mpc_response,
    plot_time_response,
    plot_ukf_estimation,
)
from helpers.simulation_runner import (
    run_custom_nonlinear_simulation,
    run_ekf_simulation,
    run_linear_simulation,
    run_mpc_simulation,
    run_ukf_simulation,
)
from helpers.system_registry import SYSTEM_REGISTRY, load_available_systems
from modules.interactive_lab import InteractiveLab, pendulum_lqr_controller


class PyControlsApp:
    """
    Main Application Controller for the PyControls Engineering Suite.

    This class handles the lifecycle of the application, including:
    1. User interaction via a CLI menu.
    2. Management of system parameters and configurations.
    3. Orchestration of various simulation modes (Time-domain, Frequency analysis,
       Parameter estimation, Nonlinear estimation, MPC, and Interactive Labs).
    """

    def __init__(self):
        """
        Initialize the application, load systems, and set up default parameters.

        It attempts to load available systems from the registry. If successful,
        it initializes the default system (DC Motor), loads configuration
        parameters (PID, LQR, Simulation settings), and prepares the environment.
        """
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
        elif self.current_system_id == "thermistor" and hasattr(
            config, "THERMISTOR_PARAMS"
        ):
            self.active_params.update(config.THERMISTOR_PARAMS)
        elif self.current_system_id == "battery" and hasattr(config, "BATTERY_PARAMS"):
            self.active_params.update(config.BATTERY_PARAMS)

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
        """
        Clears the terminal screen using OS-specific commands.
        Uses 'cls' for Windows (nt) and 'clear' for Unix-like systems.
        """
        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self):
        """
        Prints the application banner and current system active parameters.
        Truncates the parameter string if it exceeds display width to maintain layout.
        """
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
        """
        Displays the main menu loop and handles user input routing.

        Provides access to:
        - Standard Simulations
        - System Analysis (Bode/Poles)
        - Parameter Editing
        - Advanced Demos (EKF, UKF, MPC)
        - Interactive Labs
        """
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
            print("[9] Interactive Lab (Supports Hardware HIL)")
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
                self.run_ekf()
            elif choice == "7":
                self.run_ukf()
            elif choice == "8":
                self.run_mpc()
            elif choice == "9":
                self.run_interactive_lab()
            elif choice == "10":
                self.run_custom_simulation()
            elif choice == "q":
                self.running = False
            else:
                input("Invalid option. Press Enter...")

    def run_preset_dashboard(self):
        """
        Executes Option 1: Time-Domain Response Dashboard.

        Instantiates the current system and loops through defined controllers
        (e.g., PID presets or LQR) to generate comparative plots.

        Calculates and displays Step Response Metrics (Rise Time, Overshoot, Settling Time)
        in the terminal for each controller.

        Visualizations:
        - 2x2 Subplot grid.
        - Motor: Speed, Current, Voltage, Estimated Disturbance (vs True Load).
        - Pendulum: Angle, Cart Position, Force, Angular Velocity.
        """
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
            loop_controllers = [{"name": "LQR Stabilizer", "color": "k"}]

        y_real_hist = {}
        x_est_hist = {}
        u_hist_map = {}
        t = 0

        for ctrl in loop_controllers:
            t, y_real, x_est, u_hist = run_linear_simulation(
                current_system,
                self.current_system_id,
                ctrl,
                self.sim_params,
                self.dist_params,
            )

            if len(y_real) > 0:
                y_real_hist[ctrl["name"]] = y_real
                x_est_hist[ctrl["name"]] = x_est
                u_hist_map[ctrl["name"]] = u_hist

        plot_time_response(
            t,
            y_real_hist,
            x_est_hist,
            u_hist_map,
            labels,
            indices,
            loop_controllers,
            self.current_system_id,
            self.dist_params,
        )

    def run_analysis_dashboard(self):
        """
        Executes Option 2: System Analysis.

        Performs linear analysis on the State Space model.

        Calculates and displays Stability Margins (Gain Margin, Phase Margin)
        in the terminal.

        Visualizations:
        1. Bode Plot: Frequency response from Input -> Primary Output (Speed/Angle).
        2. Pole-Zero Map: Visualizes system stability in the S-Plane.
        3. Kalman Estimate Error: Difference between True State and Estimated State.
        4. Phase Plane Trajectory:
           - Pendulum: Theta vs Theta_dot.
           - Motor: Speed vs Current.
        """
        if not self.current_descriptor.supports_analysis:
            print("\nAnalysis & metrics are only available for linear systems.")
            input("Press Enter to return to menu...")
            return

        print(f"\nGenerating Analysis for {self.system_name}...")
        try:
            current_system = self.SystemClass(**self.active_params)
            ss = current_system.get_state_space()
        except Exception:
            print("System does not support state-space analysis.")
            return

        print("-" * 60)
        tf_ol = None
        ctrl_name = "Unknown"

        if self.current_system_id == "dc_motor":
            print("Stability Analysis (Loop Gain with Default PID)")
            pid_cfg = self.controllers[2]
            ctrl_name = pid_cfg["name"]
            try:
                tf_ol = current_system.get_open_loop_tf(
                    pid_cfg["Kp"], pid_cfg["Ki"], pid_cfg["Kd"]
                )
            except Exception as e:
                print(f"Could not construct TF: {e}")

        elif self.current_system_id == "pendulum":
            print("Stability Analysis (LQR Loop Gain)")
            print("Note: Margins represent robustness at the plant input.")
            ctrl_name = "LQR"
            try:
                K_lqr = current_system.dlqr_gain()
                tf_ol = current_system.get_open_loop_tf(K_lqr)
            except Exception as e:
                print(f"Could not construct TF: {e}")

        if tf_ol is not None:
            try:
                gm, pm, w_pc, w_gc = get_stability_margins(tf_ol)

                print(f"Controller: {ctrl_name}")
                print(f"  Gain Margin:     {gm:.2f} dB")
                print(f"  Phase Margin:    {pm:.2f} deg")
                print(f"  Phase Crossover: {w_pc:.2f} rad/s")
                print(f"  Gain Crossover:  {w_gc:.2f} rad/s")
            except Exception as e:
                print(f"Could not calculate margins: {e}")
        else:
            print("Margin analysis unavailable for this configuration.")

        print("-" * 60)

        w = np.logspace(*config.PLOT_PARAMS["bode_range"])
        out_idx = 2 if self.current_system_id == "pendulum" else 0
        mags, phases = ss.get_frequency_response(w, input_idx=0, output_idx=out_idx)

        t, y_real, x_est, _ = run_linear_simulation(
            current_system,
            self.current_system_id,
            self.controllers[1],
            self.sim_params,
            self.dist_params,
        )

        plot_analysis_dashboard(
            ss,
            w,
            mags,
            phases,
            t,
            y_real,
            x_est,
            self.current_system_id,
        )

    def edit_params_menu(self):
        """
        Provides an interactive CLI to edit the physical parameters of the current system.
        Updates self.active_params in real-time.
        """
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
        """
        Provides a menu to configure external disturbance injection.
        Allows toggling the disturbance on/off and setting magnitude/start time.
        """
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
            except Exception as e:
                print("Error occurred!", e, sep="\n")
        elif choice == "3":
            try:
                self.dist_params["time"] = float(input("New Time: "))
            except Exception as e:
                print("Error occurred!", e, sep="\n")

    def switch_system_menu(self):
        """
        Displays a menu to switch between available dynamic systems (e.g., DC Motor, Pendulum).

        Updates the internal state (current_system_id, active_params, SystemClass)
        based on the user's selection from the SYSTEM_REGISTRY.
        """
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
                        elif new_id == "thermistor" and hasattr(
                            config, "THERMISTOR_PARAMS"
                        ):
                            self.active_params.update(config.THERMISTOR_PARAMS)
                        elif new_id == "battery" and hasattr(config, "BATTERY_PARAMS"):
                            self.active_params.update(config.BATTERY_PARAMS)

                        self.system.params = self.active_params.copy()
                    else:
                        self.system = None
                        self.active_params = {}

                    print(f"\nSwitched to {self.current_descriptor.display_name}")
                    time.sleep(1)
                    self.clear_screen()
        except ValueError:
            print("Invalid option!")

    def run_ekf(self):
        """
        Runs the Extended Kalman Filter (EKF) demo for Joint State & Parameter Estimation.

        This demo augments the state vector to include system parameters (unknowns)
        and estimates them over time using the EKF.

        Configurations:
        1. DC Motor:
           - Estimates: Inertia (J) and Friction (b).
           - State Vector: [Speed, Current, log(J), log(b)].
           - Measurements: [Speed, Current].

        2. Pendulum:
           - Estimates: Mass (m) and Length (l).
           - State Vector: [x, v, theta, omega, log(m), log(l)].
           - Measurements: [x, theta] (Positions only).

        The simulation creates a 'True' system with predefined parameters and an
        'Estimator' initialized with guesses. It plots the convergence of the
        parameters over time.
        """

        if not self.current_descriptor.supports_estimation:
            print("\nEstimation is not supported for this system.")
            input("Press Enter to return to menu...")
            return

        print(f"\n--- Parameter Estimation Demo ({self.system_name}) ---")

        if self.current_system_id == "dc_motor":
            est_cfg = config.ESTIMATION_PARAMS
            param_names = ["Inertia (J)", "Friction (b)"]
            plot_labels = ["Speed (rad/s)", "Current (A)"]

        elif self.current_system_id == "pendulum":
            est_cfg = config.PENDULUM_ESTIMATION_PARAMS
            param_names = ["Mass (m)", "Length (l)"]
            plot_labels = ["Position (m)", "Angle (rad)"]

        t_vals, history, true_params, param_keys = run_ekf_simulation(
            self.SystemClass,
            self.current_system_id,
            est_cfg,
        )
        true_vals_list = [true_params[k] for k in param_keys]

        print("Simulating...")
        plot_estimation_history(
            t_vals, history, plot_labels, true_vals_list, param_names
        )

    def run_ukf(self):
        """
        Runs the Unscented Kalman Filter (UKF) demo for Non-linear State Estimation.

        This demo uses the Sigma-Point transform to handle non-linearities in dynamics
        or measurement models.

        Scenarios:
        1. DC Motor: Sine wave input to test estimation through zero-crossings (stiction).
           - Measures both Speed and Current.
        2. Pendulum: Free swing dynamics.
           - Measures Angle, but estimates Angular Velocity via the filter.

        Plots comparison between True State, Noisy Measurements, and UKF Estimate.
        """

        if not hasattr(self.system, "get_nonlinear_dynamics"):
            print(f"\nUKF is not supported/implemented for {self.system_name}.")
            input("Press Enter to return to menu...")
            return

        if self.system:
            self.system.params = self.active_params.copy()

        print(f"\n--- Non-Linear State Estimation (UKF) - {self.system_name} ---")

        if self.current_system_id == "pendulum":
            cfg = config.UKF_PENDULUM_PARAMS
            labels = ["Angle (rad)", "Velocity (rad/s)"]
        elif self.current_system_id == "dc_motor":
            cfg = config.UKF_MOTOR_PARAMS
            labels = ["Speed (rad/s)", "Current (A)"]
        else:
            print("Unknown system config.")
            return

        t_vals, true_states, est_states, measurements = run_ukf_simulation(
            self.system,
            self.current_system_id,
            cfg,
        )

        plot_ukf_estimation(
            t_vals,
            true_states,
            est_states,
            measurements,
            labels,
        )

    def run_mpc(self):
        """
        Runs the Model Predictive Control (MPC) demo.

        Demonstrates optimal control over a finite horizon.

        Algorithms:
        1. DC Motor: Uses Linear MPC (ADMM solver).
           - Goal: Track a target speed while minimizing voltage spikes.
        2. Pendulum: Uses Nonlinear MPC (iLQR solver).
           - Goal: Swing up from the Down position (Pi) to the Up position (0).

        Plots the system output against the reference, showing the optimized control inputs.
        """

        if not hasattr(self.system, "get_mpc_model"):
            print(f"\nMPC is not supported for {self.system_name}.")
            input("Press Enter to return to menu...")
            return

        if self.system:
            self.system.params = self.active_params.copy()

        print(f"\n--- Model Predictive Control (MPC) - {self.system_name} ---")

        if self.current_system_id == "dc_motor":
            plot_labels = ["Speed (rad/s)", "Voltage (V)"]
            cfg = config.MPC_MOTOR_PARAMS
        elif self.current_system_id == "pendulum":
            plot_labels = ["Angle (rad)", "Force (N)"]
            cfg = config.MPC_PENDULUM_PARAMS
        else:
            return

        t_vals, x_hist, u_hist, ref, plot_idx = run_mpc_simulation(
            self.system,
            self.current_system_id,
            cfg,
        )

        plot_mpc_response(
            t_vals,
            x_hist,
            u_hist,
            ref,
            plot_labels,
            plot_idx,
            self.current_system_id,
            cfg,
        )

    def run_interactive_lab(self):
        """
        Runs the Interactive Laboratory mode.

        This mode provides a real-time (or quasi-real-time) simulation loop where
        users can interact with the system or observe automatic controllers.

        Features:
        - DC Motor: Manual control or Auto-PID speed control.
        - Pendulum: Manual control or LQR stabilization (if supported).
        - Visualization: Updates a textual UI with current state values (headless mode).
        - Estimator: Can run an EKF in parallel if configured.
        """
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

        elif self.current_system_id == "pendulum":
            current_sys_instance = self.SystemClass(**self.active_params)

            if hasattr(current_sys_instance, "dlqr_gain"):
                K = current_sys_instance.dlqr_gain()
                lab.set_auto_controller(pendulum_lqr_controller(K))
            else:
                print("[WARN] System does not support LQR. Auto-control disabled.")

            x0 = np.array(config.INTERACTIVE_LAB_PARAMS["ekf_x0"])
            Q = np.diag(config.INTERACTIVE_LAB_PARAMS["ekf_Q_diag"])
            R = np.diag(config.INTERACTIVE_LAB_PARAMS["ekf_R_diag"])

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

        elif self.current_system_id == "battery":
            pid = PIDController(
                Kp=BATTERY_PID["Kp"],
                Ki=BATTERY_PID["Ki"],
                Kd=BATTERY_PID["Kd"],
                output_limits=(0, 255),
            )

            ss = self.system.get_state_space()
            phi = np.array([[np.exp(ss.A[0, 0] * lab.dt)]])
            gamma = (phi - 1) * (1 / ss.A[0, 0]) * ss.B

            kf = KalmanFilter(
                phi,
                gamma,
                ss.C,
                Q=np.array([BATTERY_KF["Q"]]),
                R=np.array([BATTERY_KF["R"]]),
                x0=[0.0],
            )
            lab.set_estimator(kf)

            def battery_ctrl(state_est, t):
                u_raw = pid.update(state_est[0], self.active_params["setpoint"], lab.dt)
                pwm = np.clip(u_raw, 0, 255)
                return pwm

            lab.set_auto_controller(battery_ctrl)

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
                elif lab.descriptor.system_id == "thermistor":
                    T = lab.state[0]
                    u = actual_u
                    print(f"\rT={T:.2f}°C, u={u:.0f}/255", end="")

                elif lab.descriptor.system_id == "battery":
                    v_est = lab.state_est[0]
                    u = lab.last_u
                    v_avg = lab.last_avg

                    print(
                        f"\rV_est: {v_est:.2f}V | PWM: {u:.0f} | V_avg: {v_avg:.2f}",
                        end="",
                    )

                time.sleep(lab.dt if not lab.descriptor.is_hardware else lab.dt * 2)

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

        input("\nPress Enter to return to menu...")

    def run_custom_simulation(self):
        """
        Runs an adaptive step-size solver (RK45) on user-defined non-linear equations.

        Prompts the user for a differential equation string (e.g., "sin(t) - x").
        Uses `make_system_func` to parse the string into a callable and solves it
        using the NonlinearSolver with adaptive time stepping.
        """
        print("\n--- Custom Non-Linear Simulation (Adaptive RK45) ---")
        eqn = input("Enter dx/dt = f(t, x, u): ").strip()
        try:
            t_vals, y_vals = run_custom_nonlinear_simulation(
                eqn,
                config.CUSTOM_SIM_PARAMS,
            )

            plot_custom_simulation(
                t_vals,
                y_vals,
                eqn,
            )

        except Exception as e:
            print(f"\nError: {e}")


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
