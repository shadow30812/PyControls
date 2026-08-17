"""
Benchmark 2.2 — MPC Solve Times.

Measures average and worst-case solve times for:
  - ADMM (DC Motor linear system)
  - iLQR (Inverted Pendulum nonlinear system)
across the configured horizons and dimensions.

Expected result: ADMM < 1ms for small systems; iLQR 1–10ms depending
on nonlinearity.

Run from the project root:
    python -m benchmarking.benchmark_mpc
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.mpc import ModelPredictiveControl
from systems.dc_motor import DCMotor
from systems.pendulum import InvertedPendulum


def benchmark_admm():
    """Benchmarks the ADMM solver on the DC Motor (linear MPC)."""
    motor = DCMotor()
    dt = 0.05
    A_d, B_d = motor.get_mpc_model(dt)

    horizons = [10, 20, 30, 50]
    n_trials = 200

    print("  --- ADMM (DC Motor, Linear MPC) ---")
    print(f"  {'Horizon':>8}  {'Mean (ms)':>10}  {'Max (ms)':>10}  {'Min (ms)':>10}")
    print("  " + "-" * 42)

    for H in horizons:
        mpc = ModelPredictiveControl(
            A=A_d,
            B=B_d,
            horizon=H,
            dt=dt,
            Q=np.diag([20.0, 0.0]),
            R=np.array([[0.01]]),
            u_min=-12.0,
            u_max=12.0,
        )

        x_current = np.array([0.0, 0.0])
        x_ref = np.array([2.5, 0.0])

                
        mpc.optimize(x_current, x_ref)

        times = []
        for _ in range(n_trials):
                                                  
            x_test = x_current + np.random.randn(2) * 0.1
            t0 = time.perf_counter_ns()
            mpc.optimize(x_test, x_ref)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            times.append(elapsed_ms)

        times = np.array(times)
        print(
            f"  {H:>8}  "
            f"{np.mean(times):>10.3f}  "
            f"{np.max(times):>10.3f}  "
            f"{np.min(times):>10.3f}"
        )

    print()


def benchmark_ilqr():
    """Benchmarks the iLQR solver on the Inverted Pendulum (nonlinear MPC)."""
    pend = InvertedPendulum()
    dt = 0.02

    model_func = pend.get_mpc_model(dt)

    horizons = [10, 20, 50]
    n_trials = 50                                     

    print("  --- iLQR (Inverted Pendulum, Nonlinear MPC) ---")
    print(f"  {'Horizon':>8}  {'Mean (ms)':>10}  {'Max (ms)':>10}  {'Min (ms)':>10}")
    print("  " + "-" * 42)

    for H in horizons:
        mpc = ModelPredictiveControl(
            model_func=model_func,
            x0=np.array([0.0, 0.0, np.pi, 0.0]),
            horizon=H,
            dt=dt,
            Q=np.diag([1.0, 0.1, 20.0, 0.1]),
            R=np.array([[0.1]]),
            u_min=-20.0,
            u_max=20.0,
        )

        x_current = np.array([0.0, 0.0, np.pi, 0.0])
        x_ref = np.array([0.0, 0.0, 0.0, 0.0])

                
        mpc.optimize(x_current, x_ref)

        times = []
        for _ in range(n_trials):
            x_test = x_current + np.random.randn(4) * 0.01
            t0 = time.perf_counter_ns()
            mpc.optimize(x_test, x_ref)
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            times.append(elapsed_ms)

        times = np.array(times)
        print(
            f"  {H:>8}  "
            f"{np.mean(times):>10.3f}  "
            f"{np.max(times):>10.3f}  "
            f"{np.min(times):>10.3f}"
        )

    print()


def benchmark_mpc():
    """Runs the full MPC solve-time benchmark."""
    print("=" * 60)
    print("Benchmark 2.2: MPC Solve Times")
    print("=" * 60)
    benchmark_admm()
    benchmark_ilqr()


if __name__ == "__main__":
    benchmark_mpc()
