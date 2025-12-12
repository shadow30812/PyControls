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
from core.math_utils import make_system_func
from core.solver import ExactSolver, NonlinearSolver


# --- Dynamic System Loader ---
def load_available_systems():
    """
    Scans the 'systems' package for modules and classes that implement
    the expected interface (get_closed_loop_tf, get_disturbance_tf).
    """
    systems = {}
    systems_path = os.path.join(os.getcwd(), "systems")

    # 1. Iterate over all files in systems/
    for _, name, _ in pkgutil.iter_modules([systems_path]):
        module_name = f"systems.{name}"
        try:
            module = importlib.import_module(module_name)

            # 2. Inspect module for classes
            for member_name, member_obj in inspect.getmembers(module, inspect.isclass):
                # Duck Typing: Check if it has the required methods
                if (
                    hasattr(member_obj, "get_closed_loop_tf")
                    and hasattr(member_obj, "get_disturbance_tf")
                    and member_obj.__module__ == module_name
                ):  # Ensure it's defined here, not imported
                    systems[member_name] = member_obj
        except Exception as e:
            print(f"Warning: Could not load system '{name}': {e}")

    return systems


class PyControlsApp:
    def __init__(self):
        # 1. Discover Systems
        self.available_systems = load_available_systems()
        if not self.available_systems:
            print("Error: No valid system classes found in systems/ folder.")
            sys.exit(1)

        # 2. Select Default System (Prefer DCMotor if available, else first found)
        self.system_name = (
            "DCMotor"
            if "DCMotor" in self.available_systems
            else next(iter(self.available_systems))
        )
        self.SystemClass = self.available_systems[self.system_name]

        # 3. Load Active Parameters
        # We instantiate a temporary object to get its default 'self.params'
        temp_instance = self.SystemClass()
        self.active_params = temp_instance.params.copy()

        # Apply Config Overrides (Backward Compatibility for DCMotor)
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

        # Dynamic Parameter Printing
        # Formats params into a readable string, truncating if too long
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
            print("[1] Run Standard Simulation with Bode Plot (Linear - Exact ZOH)")
            print("[2] Run Custom Non-Linear Simulation (Adaptive RK45)")
            print("[3] Edit System Parameters")
            print("[4] Edit Disturbance Settings")
            print("[5] Switch Active System")
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
            elif choice == "q":
                self.running = False
                self.clear_screen()
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

                    # Reset params to the new system's defaults
                    self.active_params = self.SystemClass().params.copy()

                    # Optional: Re-apply config if specific overrides exist
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
        Kp, Ki, Kd = ctrl_config["Kp"], ctrl_config["Ki"], ctrl_config["Kd"]

        # Polymorphic Calls: These work for ANY system class found in /systems
        tf_ref = system_instance.get_closed_loop_tf(Kp, Ki, Kd)
        tf_dist = system_instance.get_disturbance_tf(Kp, Ki, Kd)

        Ar, Br, Cr, Dr = tf_ref.to_state_space()
        solver_ref = ExactSolver(Ar, Br, Cr, Dr, dt=dt)

        Ad, Bd, Cd, Dd = tf_dist.to_state_space()
        solver_dist = ExactSolver(Ad, Bd, Cd, Dd, dt=dt)

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
        print(f"\nInitializing Simulation for {self.system_name}...")

        # Instantiate the currently selected system dynamically
        try:
            current_system = self.SystemClass(**self.active_params)
        except Exception as e:
            print(f"Error instantiating {self.system_name}: {e}")
            return

        fig, axes = plt.subplots(1, 2, figsize=config.PLOT_PARAMS["figsize"])
        ax_time, ax_bode = axes[0], axes[1]

        # Header for the console output table
        print("-" * 90)
        print(
            f"{'Controller':<20} | {'Rise Time':<10} | {'Overshoot':<10} | {'GM (dB)':<10} | {'PM (deg)':<10}"
        )
        print("-" * 90)

        for ctrl in self.controllers:
            t, y = self.simulate_preset_system(current_system, ctrl)

            # Metrics Calculation
            try:
                dist_index = np.searchsorted(t, self.dist_params["time"])
                # Handle edge case where disturbance is beyond simulation time
                if dist_index >= len(t) or dist_index == 0:
                    dist_index = len(t)
                metrics = get_step_metrics(t[:dist_index], y[:dist_index])
            except Exception:
                metrics = (0, 0, 0)

            # Stability Margins
            tf = current_system.get_closed_loop_tf(ctrl["Kp"], ctrl["Ki"], ctrl["Kd"])
            gm, pm, _, _ = get_stability_margins(tf)

            print(
                f"{ctrl['name']:<20} | {metrics[0]:.4f}s    | {metrics[1]:.2f}%      | {gm:.2f}       | {pm:.2f}"
            )

            ax_time.plot(t, y, label=ctrl["name"], color=ctrl["color"])

            # Bode Plot
            w = np.logspace(*config.PLOT_PARAMS["bode_range"])
            mags, _ = tf.bode_response(w)
            ax_bode.semilogx(w, mags, label=ctrl["name"], color=ctrl["color"])

        # Plot Styling
        if self.dist_params["enabled"]:
            d_time = self.dist_params["time"]
            style = config.PLOT_PARAMS["marker_style"]
            ax_time.axvline(x=d_time, **style)
            ax_time.text(d_time + 0.05, 0.1, "Disturbance", fontsize=9)

        ax_time.set_title(f"Step Response ({self.system_name})")
        ax_time.grid(True, alpha=0.3)
        ax_time.legend()
        ax_bode.set_title("Bode Magnitude")
        ax_bode.grid(True, alpha=0.3)

        print("\nPlot generated. Close window to return to menu.")
        plt.tight_layout()
        plt.show()

    def run_custom_simulation(self):
        print("\n--- Custom Non-Linear Simulation (Adaptive RK45) ---")
        eqn = input("Enter dx/dt = f(t, x, u): ").strip()
        try:
            dyn_func = make_system_func(eqn)

            x0 = np.zeros(config.CUSTOM_SIM_PARAMS["initial_state"]).flatten()

            # Use the new NonlinearSolver with adaptive steps
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


if __name__ == "__main__":
    app = PyControlsApp()
    try:
        app.main_menu()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        time.sleep(2)
    finally:
        app.clear_screen()
