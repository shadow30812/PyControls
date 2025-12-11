import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Import local modules
sys.path.append(os.getcwd())
import config
from core.analysis import get_stability_margins, get_step_metrics
from core.math_utils import make_system_func
from core.solver import RK4Solver
from systems.dc_motor import DCMotor


def simulate_preset_system(motor, ctrl_config):
    """Simulates Linear System from Config."""
    dt = config.SIM_PARAMS["dt"]
    t_end = config.SIM_PARAMS["t_end"]
    dist_time = config.DISTURBANCE_PARAMS["time"]
    dist_mag = config.DISTURBANCE_PARAMS["magnitude"]

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
        u_ref = config.SIM_PARAMS["step_volts"]
        y_ref = solver_ref.step(u_ref)

        if config.DISTURBANCE_PARAMS["enabled"] and t >= dist_time:
            u_dist = dist_mag
        else:
            u_dist = 0.0
        y_dist = solver_dist.step(u_dist)

        y_total.append(y_ref + y_dist)

    return t_values, np.array(y_total)


def run_preset_dashboard():
    """Runs the standard comparative dashboard."""
    motor = DCMotor(**config.MOTOR_PARAMS)
    print(f"Loaded Motor: {config.MOTOR_PARAMS}")

    fig, axes = plt.subplots(1, 2, figsize=config.PLOT_PARAMS["figsize"])
    ax_time, ax_bode = axes[0], axes[1]

    print("-" * 90)
    print(
        f"{'Controller':<20} | {'Rise Time':<10} | {'Overshoot':<10} | {'GM (dB)':<10} | {'PM (deg)':<10}"
    )
    print("-" * 90)

    for ctrl in config.CONTROLLERS:
        t, y = simulate_preset_system(motor, ctrl)
        dist_index = np.searchsorted(t, config.DISTURBANCE_PARAMS["time"])
        metrics = get_step_metrics(t[:dist_index], y[:dist_index])

        tf = motor.get_closed_loop_tf(ctrl["Kp"], ctrl["Ki"], ctrl["Kd"])
        gm, pm, _, _ = get_stability_margins(tf)

        print(
            f"{ctrl['name']:<20} | {metrics[0]:.4f}s    | {metrics[1]:.2f}%      | {gm:.2f}       | {pm:.2f}"
        )

        ax_time.plot(t, y, label=ctrl["name"], color=ctrl["color"], linewidth=2)

        w_start, w_end, w_pts = config.PLOT_PARAMS["bode_range"]
        w = np.logspace(w_start, w_end, w_pts)
        mags, _ = tf.bode_response(w)
        ax_bode.semilogx(w, mags, label=ctrl["name"], color=ctrl["color"], linewidth=2)

    if config.DISTURBANCE_PARAMS["enabled"]:
        d_time = config.DISTURBANCE_PARAMS["time"]
        style = config.PLOT_PARAMS["marker_style"]
        text_cfg = config.PLOT_PARAMS["marker_text"]
        ax_time.axvline(x=d_time, **style)
        ax_time.text(
            d_time + text_cfg["x_offset"],
            text_cfg["y_pos"],
            text_cfg["label"],
            fontsize=text_cfg["fontsize"],
        )

    ax_time.set_title("Step Response + Disturbance Rejection")
    ax_time.grid(True, alpha=config.PLOT_PARAMS["grid_alpha"])
    ax_time.legend()
    ax_bode.set_title("Bode Magnitude")
    ax_bode.grid(True, alpha=config.PLOT_PARAMS["grid_alpha"])
    plt.tight_layout()
    plt.show()


def run_custom_simulation():
    """Interactive mode for user-defined equations."""
    print("\n--- Custom Non-Linear Simulation ---")
    print("Define your system differential equation: dx/dt = f(x, u)")
    print("Variables: 'x' (state vector), 'u' (scalar input).")

    eqn = input("Enter dx/dt equation (e.g. -x + sin(u)): ").strip()

    try:
        dyn_func = make_system_func(eqn)

        dt = config.CUSTOM_SIM_PARAMS["dt"]
        t_end = config.CUSTOM_SIM_PARAMS["t_end"]
        step_t = config.CUSTOM_SIM_PARAMS["step_time"]
        step_mag = config.CUSTOM_SIM_PARAMS["step_magnitude"]
        init_shape = config.CUSTOM_SIM_PARAMS["initial_state"]

        # NOTE: This requires the Hybrid Solver!
        solver = RK4Solver(dt=dt, dynamics_func=dyn_func)
        solver.x = np.zeros(init_shape)

        t_values = np.linspace(0, t_end, int(t_end / dt))
        y_values = []

        print("Simulating...")
        for t in t_values:
            u = step_mag if t > step_t else 0.0
            y = solver.step(u)
            y_values.append(y)

        plt.figure(figsize=config.PLOT_PARAMS["figsize"])
        plt.plot(t_values, y_values, label=f"dx/dt = {eqn}")
        plt.title(f"Custom Simulation: {eqn}")
        plt.xlabel("Time (s)")
        plt.ylabel("State x")
        plt.grid(True, alpha=config.PLOT_PARAMS["grid_alpha"])
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Welcome to PyControls")
    print("1. Run Preset Config Dashboard (Linear DC Motor)")
    print("2. Run Custom Equation Simulation (Non-Linear)")

    choice = input("Select Mode (1/2): ").strip()

    if choice == "2":
        run_custom_simulation()
    else:
        run_preset_dashboard()
