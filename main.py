import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Import local modules
sys.path.append(os.getcwd())
import config
from core.analysis import get_stability_margins, get_step_metrics
from core.math_utils import make_system_func
from core.solver import RK4Solver
from systems.dc_motor import DCMotor


class PyControlsApp:
    def __init__(self):
        # Load Config into Runtime Memory (so we can edit it)
        self.motor_params = config.MOTOR_PARAMS.copy()
        self.controllers = config.CONTROLLERS.copy()
        self.sim_params = config.SIM_PARAMS.copy()
        self.dist_params = config.DISTURBANCE_PARAMS.copy()
        self.running = True

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self):
        # Only print the info block, do NOT clear screen
        print("\n" + "=" * 60)
        print("   PyControls Engineering Suite | v1.0   ")
        print("=" * 60)
        print(
            f"Active Motor: J={self.motor_params['J']}, b={self.motor_params['b']}, K={self.motor_params['K']}"
        )
        print(
            f"Disturbance:  {'ON' if self.dist_params['enabled'] else 'OFF'} (Mag={self.dist_params['magnitude']} at t={self.dist_params['time']}s)"
        )
        print("-" * 60)

    def main_menu(self):
        # Clear screen ONLY once at startup
        self.clear_screen()

        while self.running:
            self.print_header()
            print("[1] Run Standard Simulation (Step Response + Bode)")
            print("[2] Run Custom Non-Linear Simulation (Equation)")
            print("[3] Edit Motor Parameters")
            print("[4] Edit Disturbance Settings")
            print("[q] Exit")

            choice = input("\nSelect Option: ").strip()

            if choice == "1":
                self.run_preset_dashboard()
            elif choice == "2":
                self.run_custom_simulation()
            elif choice == "3":
                self.edit_motor_menu()
            elif choice == "4":
                self.edit_disturbance_menu()
            elif choice == "q":
                self.running = False
                self.clear_screen()  # Clear on exit
                print("Goodbye!")
            else:
                input("Invalid option. Press Enter...")

    def edit_motor_menu(self):
        # No loop here, just one-shot edit to keep history visible
        print("\nCurrent Motor Parameters:")
        for k, v in self.motor_params.items():
            print(f"  [{k}] : {v}")

        key = input("\nEnter parameter key to edit (or 'b' to go back): ").strip()
        if key.lower() == "b":
            return

        if key in self.motor_params:
            try:
                val = float(input(f"Enter new value for {key}: "))
                self.motor_params[key] = val
                print(f"Updated {key} to {val}")
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
            print(f"Disturbance enabled: {self.dist_params['enabled']}")
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

    def simulate_preset_system(self, motor, ctrl_config):
        dt = self.sim_params["dt"]
        t_end = self.sim_params["t_end"]

        Kp, Ki, Kd = ctrl_config["Kp"], ctrl_config["Ki"], ctrl_config["Kd"]

        tf_ref = motor.get_closed_loop_tf(Kp, Ki, Kd)
        tf_dist = motor.get_disturbance_tf(Kp, Ki, Kd)

        Ar, Br, Cr, Dr = tf_ref.to_state_space()
        solver_ref = RK4Solver(Ar, Br, Cr, Dr, dt=dt)
        Ad, Bd, Cd, Dd = tf_dist.to_state_space()
        solver_dist = RK4Solver(Ad, Bd, Cd, Dd, dt=dt)

        t_values = np.linspace(0, t_end, int(t_end / dt))
        y_total = []

        for t in t_values:
            u_ref = self.sim_params["step_volts"]
            y_ref = solver_ref.step(u_ref)

            if self.dist_params["enabled"] and t >= self.dist_params["time"]:
                u_dist = self.dist_params["magnitude"]
            else:
                u_dist = 0.0
            y_dist = solver_dist.step(u_dist)

            y_total.append(y_ref + y_dist)

        return t_values, np.array(y_total)

    def run_preset_dashboard(self):
        print("\nInitializing Simulation...")
        motor = DCMotor(**self.motor_params)

        fig, axes = plt.subplots(1, 2, figsize=config.PLOT_PARAMS["figsize"])
        ax_time, ax_bode = axes[0], axes[1]

        print("-" * 90)
        print(
            f"{'Controller':<20} | {'Rise Time':<10} | {'Overshoot':<10} | {'GM (dB)':<10} | {'PM (deg)':<10}"
        )
        print("-" * 90)

        for ctrl in self.controllers:
            t, y = self.simulate_preset_system(motor, ctrl)

            # Safe metrics calculation
            try:
                dist_index = np.searchsorted(t, self.dist_params["time"])
                if dist_index >= len(t) or dist_index == 0:
                    dist_index = len(t)
                metrics = get_step_metrics(t[:dist_index], y[:dist_index])
            except Exception:
                metrics = (0, 0, 0)

            tf = motor.get_closed_loop_tf(ctrl["Kp"], ctrl["Ki"], ctrl["Kd"])
            gm, pm, _, _ = get_stability_margins(tf)

            print(
                f"{ctrl['name']:<20} | {metrics[0]:.4f}s    | {metrics[1]:.2f}%      | {gm:.2f}       | {pm:.2f}"
            )

            ax_time.plot(t, y, label=ctrl["name"], color=ctrl["color"], linewidth=2)

            w_start, w_end, w_pts = config.PLOT_PARAMS["bode_range"]
            w = np.logspace(w_start, w_end, w_pts)
            mags, _ = tf.bode_response(w)
            ax_bode.semilogx(
                w, mags, label=ctrl["name"], color=ctrl["color"], linewidth=2
            )

        if self.dist_params["enabled"]:
            d_time = self.dist_params["time"]
            style = config.PLOT_PARAMS["marker_style"]
            ax_time.axvline(x=d_time, **style)
            ax_time.text(d_time + 0.05, 0.1, "Disturbance", fontsize=9)

        ax_time.set_title("Step Response")
        ax_time.grid(True, alpha=0.3)
        ax_time.legend()
        ax_bode.set_title("Bode Magnitude")
        ax_bode.grid(True, alpha=0.3)

        print("\nPlot generated. Close window to return to menu.")
        plt.tight_layout()
        plt.show()

    def run_custom_simulation(self):
        print("\n--- Custom Non-Linear Simulation ---")
        print("Equations support numpy functions: sin, cos, exp, tanh, abs...")
        eqn = input("Enter dx/dt = f(x, u): ").strip()

        try:
            dyn_func = make_system_func(eqn)
            dt = config.CUSTOM_SIM_PARAMS["dt"]
            t_end = config.CUSTOM_SIM_PARAMS["t_end"]

            solver = RK4Solver(dt=dt, dynamics_func=dyn_func)
            solver.x = np.zeros(config.CUSTOM_SIM_PARAMS["initial_state"])

            t_vals = np.linspace(0, t_end, int(t_end / dt))
            y_vals = []

            print("Simulating...")
            for t in t_vals:
                u = 1.0 if t > 0.5 else 0.0
                y = solver.step(u)
                y_vals.append(y)

            plt.figure(figsize=config.PLOT_PARAMS["figsize"])
            plt.plot(t_vals, y_vals, label=f"dx/dt = {eqn}")
            plt.title(f"Custom: {eqn}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    app = PyControlsApp()
    try:
        app.main_menu()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        time.sleep(2)
    finally:
        app.clear_screen()
