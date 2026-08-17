"""
Benchmark 3.1 & 3.2 — Control Performance Metrics.

Provides:
  - calculate_step_metrics(t, y, target): Extracts Rise Time, Settling Time,
    Overshoot, and Steady-State Error from a step response.
  - calculate_disturbance_metrics(t, y, target, disturbance_time): Computes
    IAE and ITAE during disturbance recovery.
  - Simulates PID, LQR, and MPC on the DC Motor, comparing controller
    performance across all three.

Run from the project root:
    python -m benchmarking.control_metrics
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.control_utils import PIDController, dlqr
from core.mpc import ModelPredictiveControl
from core.solver import ExactSolver
from helpers.config import (
    CONTROLLERS,
    DISTURBANCE_PARAMS,
    MPC_MOTOR_PARAMS,
    MOTOR_PARAMS,
    SIM_PARAMS,
)
from systems.dc_motor import DCMotor


def calculate_step_metrics(t, y, target):
    """
    Analyzes a step response to extract classical control metrics.

    Args:
        t: Time array.
        y: Output response array.
        target: The desired steady-state target value.

    Returns:
        dict with keys: 'Rise Time (s)', 'Settling Time (s)',
                        'Overshoot (%)', 'SS Error'.
    """
    steady_state_value = y[-1]

    lower_bound = target * 0.1
    upper_bound = target * 0.9
    try:
        t_10 = t[np.where(y >= lower_bound)[0][0]]
        t_90 = t[np.where(y >= upper_bound)[0][0]]
        rise_time = t_90 - t_10
    except IndexError:
        rise_time = np.inf

    error_band = target * 0.02
    settled_indices = np.where(np.abs(y - target) > error_band)[0]
    if len(settled_indices) > 0:
        settling_time = t[settled_indices[-1]]
    else:
        settling_time = 0.0

    peak_value = np.max(y)
    if peak_value > target:
        overshoot_pct = ((peak_value - target) / target) * 100
    else:
        overshoot_pct = 0.0

    ss_error = np.abs(target - steady_state_value)

    return {
        "Rise Time (s)": rise_time,
        "Settling Time (s)": settling_time,
        "Overshoot (%)": overshoot_pct,
        "SS Error": ss_error,
    }


def calculate_disturbance_metrics(t, y, target, disturbance_time):
    """
    Calculates disturbance rejection metrics for the recovery phase.

    Args:
        t: Time array.
        y: Output response array.
        target: The desired steady-state value.
        disturbance_time: The time at which the disturbance was applied.

    Returns:
        dict with keys: 'IAE', 'ITAE', 'Peak Deviation', 'Recovery Time (s)'.
    """
    mask = t >= disturbance_time
    t_rec = t[mask]
    y_rec = y[mask]
    error_rec = np.abs(y_rec - target)

    if len(t_rec) < 2:
        return {
            "IAE": np.inf,
            "ITAE": np.inf,
            "Peak Deviation": np.inf,
            "Recovery Time (s)": np.inf,
        }

    dt = t_rec[1] - t_rec[0]

    iae = np.sum(error_rec) * dt

    t_shifted = t_rec - disturbance_time
    itae = np.sum(t_shifted * error_rec) * dt

    peak_dev = np.max(np.abs(y_rec - target))

    error_band = np.abs(target) * 0.05 if target != 0 else 0.05
    outside_band = np.where(error_rec > error_band)[0]
    if len(outside_band) > 0:
        recovery_time = t_rec[outside_band[-1]] - disturbance_time
    else:
        recovery_time = 0.0

    return {
        "IAE": iae,
        "ITAE": itae,
        "Peak Deviation": peak_dev,
        "Recovery Time (s)": recovery_time,
    }


def simulate_pid_motor(Kp, Ki, Kd, target_speed, dt, t_end, disturbance_time, dist_mag):
    """Simulates a PID-controlled DC Motor and returns (t, y)."""
    motor = DCMotor(**MOTOR_PARAMS)
    ss = motor.get_state_space()

    solver = ExactSolver(ss.A, ss.B, ss.C, ss.D, dt)
    pid = PIDController(Kp, Ki, Kd, output_limits=(-12, 12))

    n_steps = int(t_end / dt)
    t_arr = np.linspace(0, t_end, n_steps)
    y_arr = np.zeros(n_steps)

    for k in range(n_steps):
        y_out = solver.step(np.array([0.0, 0.0]))

        if isinstance(y_out, np.ndarray):
            speed = y_out[0]
        else:
            speed = float(y_out)

        y_arr[k] = speed

        u_ctrl = pid.update(speed, target_speed, dt)

        torque = dist_mag if t_arr[k] >= disturbance_time else 0.0

        solver.step(np.array([u_ctrl, torque]))

    return t_arr, y_arr


def simulate_lqr_motor(target_speed, dt, t_end, disturbance_time, dist_mag):
    """Simulates an LQR-controlled DC Motor and returns (t, y)."""
    motor = DCMotor(**MOTOR_PARAMS)
    ss = motor.get_state_space()

    A_d = np.eye(ss.A.shape[0]) + ss.A * dt
    B_d = ss.B * dt

    Q = np.diag([20.0, 0.0])
    R = np.array([[0.01]])
    K_lqr = dlqr(A_d, B_d[:, :1], Q, R)

    solver = ExactSolver(ss.A, ss.B, ss.C, ss.D, dt)

    n_steps = int(t_end / dt)
    t_arr = np.linspace(0, t_end, n_steps)
    y_arr = np.zeros(n_steps)

    x_est = np.array([0.0, 0.0])

    for k in range(n_steps):
        y_out = solver.step(np.array([0.0, 0.0]))

        if isinstance(y_out, np.ndarray):
            speed = y_out[0]
        else:
            speed = float(y_out)

        y_arr[k] = speed

        x_est = np.array([speed, x_est[1]])
        x_ref = np.array([target_speed, 0.0])
        e = x_est - x_ref
        u_ctrl = float((-K_lqr @ e).flatten()[0])
        u_ctrl = np.clip(u_ctrl, -12.0, 12.0)

        torque = dist_mag if t_arr[k] >= disturbance_time else 0.0

        solver.step(np.array([u_ctrl, torque]))

    return t_arr, y_arr


def simulate_mpc_motor(target_speed, dt_sim, t_end, disturbance_time, dist_mag):
    """Simulates an MPC-controlled DC Motor and returns (t, y)."""
    motor = DCMotor(**MOTOR_PARAMS)
    ss = motor.get_state_space()

    mpc_dt = MPC_MOTOR_PARAMS["dt"]
    A_d, B_d = motor.get_mpc_model(mpc_dt)

    mpc = ModelPredictiveControl(
        A=A_d,
        B=B_d,
        horizon=MPC_MOTOR_PARAMS["horizon"],
        dt=mpc_dt,
        Q=np.diag(MPC_MOTOR_PARAMS["Q_diag"]),
        R=np.diag(MPC_MOTOR_PARAMS["R_diag"]),
        u_min=MPC_MOTOR_PARAMS["u_min"],
        u_max=MPC_MOTOR_PARAMS["u_max"],
    )

    solver = ExactSolver(ss.A, ss.B, ss.C, ss.D, dt_sim)

    n_steps = int(t_end / dt_sim)
    t_arr = np.linspace(0, t_end, n_steps)
    y_arr = np.zeros(n_steps)

    mpc_stride = max(1, int(mpc_dt / dt_sim))
    u_ctrl = 0.0
    x_est = np.array([0.0, 0.0])
    x_ref = np.array([target_speed, 0.0])

    for k in range(n_steps):
        y_out = solver.step(np.array([0.0, 0.0]))

        if isinstance(y_out, np.ndarray):
            speed = y_out[0]
        else:
            speed = float(y_out)

        y_arr[k] = speed

        x_est = np.array([speed, x_est[1]])

        if k % mpc_stride == 0:
            u_opt = mpc.optimize(x_est, x_ref)
            u_ctrl = float(np.clip(u_opt.flatten()[0],
                                   MPC_MOTOR_PARAMS["u_min"],
                                   MPC_MOTOR_PARAMS["u_max"]))

        torque = dist_mag if t_arr[k] >= disturbance_time else 0.0

        solver.step(np.array([u_ctrl, torque]))

    return t_arr, y_arr


def _print_step_table(results):
    """Prints the step response metrics table."""
    print(
        f"  {'Controller':<18} {'Rise (s)':>8} {'Settle (s)':>10} {'OS (%)':>8} {'SS Err':>8}"
    )
    print("  " + "-" * 56)
    for name, metrics in results:
        print(
            f"  {name:<18} "
            f"{metrics['Rise Time (s)']:>8.4f} "
            f"{metrics['Settling Time (s)']:>10.4f} "
            f"{metrics['Overshoot (%)']:>8.2f} "
            f"{metrics['SS Error']:>8.5f}"
        )


def _print_disturbance_table(results):
    """Prints the disturbance rejection metrics table."""
    print(
        f"  {'Controller':<18} {'IAE':>10} {'ITAE':>10} {'Peak Dev':>10} {'Recovery (s)':>12}"
    )
    print("  " + "-" * 64)
    for name, dist_metrics in results:
        print(
            f"  {name:<18} "
            f"{dist_metrics['IAE']:>10.4f} "
            f"{dist_metrics['ITAE']:>10.4f} "
            f"{dist_metrics['Peak Deviation']:>10.4f} "
            f"{dist_metrics['Recovery Time (s)']:>12.4f}"
        )


def demo_control_metrics():
    """Demonstrates the control metrics on PID, LQR, and MPC controllers."""
    target_speed = SIM_PARAMS["step_volts"]
    dt = SIM_PARAMS["dt"]
    t_end = SIM_PARAMS["t_end"]
    dist_time = DISTURBANCE_PARAMS["time"]
    dist_mag = DISTURBANCE_PARAMS["magnitude"]

    simulations = []

    for ctrl in CONTROLLERS:
        t, y = simulate_pid_motor(
            ctrl["Kp"], ctrl["Ki"], ctrl["Kd"],
            target_speed, dt, t_end, dist_time, dist_mag,
        )
        simulations.append((ctrl["name"], t, y))

    t_lqr, y_lqr = simulate_lqr_motor(target_speed, dt, t_end, dist_time, dist_mag)
    simulations.append(("LQR", t_lqr, y_lqr))

    t_mpc, y_mpc = simulate_mpc_motor(target_speed, dt, t_end, dist_time, dist_mag)
    simulations.append(("MPC (ADMM)", t_mpc, y_mpc))

    print("=" * 60)
    print("Benchmark 3.1: Step Response Metrics")
    print("=" * 60)
    print(f"\n  Target Speed: {target_speed} rad/s")
    print(f"  Simulation:   {t_end}s at dt={dt}s")
    print(f"  Disturbance:  {dist_mag} Nm at t={dist_time}s")
    print()

    step_results = []
    for name, t, y in simulations:
        step_results.append((name, calculate_step_metrics(t, y, target_speed)))
    _print_step_table(step_results)

    print()
    print("=" * 60)
    print("Benchmark 3.2: Disturbance Rejection Metrics")
    print("=" * 60)
    print()

    dist_results = []
    for name, t, y in simulations:
        dist_results.append((name, calculate_disturbance_metrics(t, y, target_speed, dist_time)))
    _print_disturbance_table(dist_results)

    print()


if __name__ == "__main__":
    demo_control_metrics()
