"""
Benchmark 2.3 — EKF/UKF Per-Step Timing.

Measures execution time for a single predict+update cycle of the Extended
Kalman Filter and Unscented Kalman Filter on the DC Motor system.

Expected result: Sub-millisecond execution times. UKF slower than EKF
by a constant factor due to sigma point generation/propagation.

Run from the project root:
    python -m benchmarking.benchmark_ekf_ukf_timing
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ekf import ExtendedKalmanFilter
from core.ukf import UnscentedKalmanFilter
from helpers.config import UKF_MOTOR_PARAMS
from systems.dc_motor import DCMotor


def benchmark_ekf_timing():
    """Benchmarks EKF predict+update cycle on the DC Motor."""
    motor = DCMotor()
    f_dynamics = motor.get_parameter_estimation_func()

    def h_meas(x):
        return np.array([x[0], x[1]])

    dt = 0.001

    ekf = ExtendedKalmanFilter(
        f_dynamics=f_dynamics,
        h_measurement=h_meas,
        Q=np.diag([1e-4, 1e-4, 2e-4, 2e-4]),
        R=np.diag([0.01, 0.01]),
        x0=np.array([0.0, 0.0, np.log(0.005), np.log(0.1)]),
        p_init_scale=0.1,
    )

    n_trials = 5000
    u_val = 5.0
    y_meas = np.array([[1.0], [0.5]])

    ekf.predict(u_val, dt)
    ekf.update(y_meas)

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter_ns()
        ekf.predict(u_val, dt)
        ekf.update(y_meas)
        elapsed_us = (time.perf_counter_ns() - t0) / 1e3
        times.append(elapsed_us)

    return np.array(times)


def benchmark_ukf_timing():
    """Benchmarks UKF predict+update cycle on the DC Motor."""
    motor = DCMotor()
    f_dynamics, h_meas = motor.get_nonlinear_dynamics()

    n_states = 2
    dt = UKF_MOTOR_PARAMS["dt"]

    ukf = UnscentedKalmanFilter(
        f_dynamics=f_dynamics,
        h_measurement=h_meas,
        Q=np.diag(UKF_MOTOR_PARAMS["Q_diag"]),
        R=np.diag(UKF_MOTOR_PARAMS["R_diag"]),
        x0=np.array(UKF_MOTOR_PARAMS["x0"]),
        P0=np.eye(n_states) * UKF_MOTOR_PARAMS["P0"],
        alpha=UKF_MOTOR_PARAMS["alpha"],
        beta=UKF_MOTOR_PARAMS["beta"],
        kappa=UKF_MOTOR_PARAMS["kappa"],
    )

    n_trials = 5000
    u_val = 5.0
    z_meas = np.array([1.0, 0.5])

    ukf.predict(u_val, dt)
    ukf.update(z_meas)

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter_ns()
        ukf.predict(u_val, dt)
        ukf.update(z_meas)
        elapsed_us = (time.perf_counter_ns() - t0) / 1e3
        times.append(elapsed_us)

    return np.array(times)


def benchmark_ekf_ukf():
    """Runs the full EKF/UKF timing benchmark."""
    print("=" * 60)
    print("Benchmark 2.3: EKF/UKF Per-Step Timing")
    print("=" * 60)

    ekf_times = benchmark_ekf_timing()
    ukf_times = benchmark_ukf_timing()

    print(f"  {'Filter':>6}  {'States':>6}  {'Mean (μs)':>10}  {'Min (μs)':>10}  {'Max (μs)':>10}  {'Median (μs)':>12}")
    print("  " + "-" * 60)
    print(
        f"  {'EKF':>6}  {4:>6}  "
        f"{np.mean(ekf_times):>10.1f}  "
        f"{np.min(ekf_times):>10.1f}  "
        f"{np.max(ekf_times):>10.1f}  "
        f"{np.median(ekf_times):>12.1f}"
    )
    print(
        f"  {'UKF':>6}  {2:>6}  "
        f"{np.mean(ukf_times):>10.1f}  "
        f"{np.min(ukf_times):>10.1f}  "
        f"{np.max(ukf_times):>10.1f}  "
        f"{np.median(ukf_times):>12.1f}"
    )

    ratio = np.mean(ekf_times) / np.mean(ukf_times)
    if ratio > 1:
        print(f"\n  UKF is {ratio:.1f}x faster than EKF on average (fewer states).")
    else:
        print(f"\n  EKF is {1/ratio:.1f}x faster than UKF on average.")

    if np.mean(ekf_times) < 1000 and np.mean(ukf_times) < 1000:
        print("  ✓ Both filters execute in sub-millisecond times per cycle.")
    print()


if __name__ == "__main__":
    benchmark_ekf_ukf()
