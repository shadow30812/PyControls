"""
Benchmark 2.4 — Algorithmic Scalability Study.

Measures computation time vs state dimension n (from 2 to 50) for:
  - Discrete Algebraic Riccati Equation (DARE) solver
  - EKF predict+update cycle
  - MPC (ADMM linear) optimize cycle

Fits a polynomial on log-log scale to verify theoretical O(n³) scaling.

Run from the project root:
    python -m benchmarking.benchmark_scalability
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.control_utils import solve_discrete_riccati
from core.ekf import ExtendedKalmanFilter
from core.mpc import ModelPredictiveControl


def _make_stable_system(n, m=1):
    """Generates a random stable discrete-time system of dimension n."""
    eigvals = np.random.uniform(0.3, 0.9, n) * np.sign(np.random.randn(n))
    V = np.random.randn(n, n)
    while np.abs(np.linalg.det(V)) < 0.1:
        V = np.random.randn(n, n)
    A = V @ np.diag(eigvals) @ np.linalg.inv(V)
    B = np.random.randn(n, m)
    return A, B


def benchmark_dare_scaling(dims, n_trials=5):
    """Measures DARE solve time vs dimension."""
    results = []
    for n in dims:
        A, B = _make_stable_system(n)
        Q = np.eye(n)
        R = np.eye(1) * 0.1

        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter_ns()
            solve_discrete_riccati(A, B, Q, R, tol=1e-8, max_iter=500)
            elapsed_us = (time.perf_counter_ns() - t0) / 1e3
            times.append(elapsed_us)

        results.append((n, np.median(times)))
    return results


def benchmark_ekf_scaling(dims, n_trials=20):
    """Measures EKF predict+update time vs dimension."""
    results = []
    for n in dims:
        A_cont = np.random.randn(n, n) * 0.1
        eigvals = np.linalg.eigvals(A_cont)
        max_real = np.max(np.real(eigvals))
        if max_real > 0:
            A_cont -= np.eye(n) * (max_real + 0.5)

        def f_dynamics(x, u, _A=A_cont):
            return _A @ x

        def h_meas(x, _n=n):
            return x[:_n]

        ekf = ExtendedKalmanFilter(
            f_dynamics=f_dynamics,
            h_measurement=h_meas,
            Q=np.eye(n) * 1e-3,
            R=np.eye(n) * 1e-2,
            x0=np.zeros(n),
            p_init_scale=0.1,
        )

        dt = 0.01
        u = 0.0
        y = np.random.randn(n, 1)

        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter_ns()
            ekf.predict(u, dt)
            ekf.update(y)
            elapsed_us = (time.perf_counter_ns() - t0) / 1e3
            times.append(elapsed_us)

        results.append((n, np.median(times)))
    return results


def benchmark_mpc_scaling(dims, n_trials=5):
    """Measures linear MPC (ADMM) optimize time vs dimension."""
    results = []
    for n in dims:
        A, B = _make_stable_system(n, m=1)
        A_d = np.eye(n) + A * 0.01

        try:
            mpc = ModelPredictiveControl(
                A=A_d,
                B=B,
                horizon=10,
                dt=0.01,
                Q=np.eye(n),
                R=np.eye(1) * 0.1,
                u_min=-10.0,
                u_max=10.0,
            )

            x0 = np.random.randn(n)
            x_ref = np.zeros(n)

            mpc.optimize(x0, x_ref)

            times = []
            for _ in range(n_trials):
                t0 = time.perf_counter_ns()
                mpc.optimize(x0, x_ref)
                elapsed_us = (time.perf_counter_ns() - t0) / 1e3
                times.append(elapsed_us)

            results.append((n, np.median(times)))
        except Exception as e:
            print(f"    MPC failed at n={n}: {e}")

    return results


def fit_scaling_exponent(results):
    """Fits log(time) = a * log(n) + b and returns the exponent a."""
    ns = np.array([r[0] for r in results], dtype=float)
    ts = np.array([r[1] for r in results], dtype=float)
    mask = ts > 0
    if np.sum(mask) < 2:
        return float("nan")
    log_n = np.log(ns[mask])
    log_t = np.log(ts[mask])
    coeffs = np.polyfit(log_n, log_t, 1)
    return coeffs[0]


def benchmark_scalability():
    """Runs the full algorithmic scalability study."""
    print("=" * 60)
    print("Benchmark 2.4: Algorithmic Scalability Study")
    print("=" * 60)

    dims_dare = [2, 3, 5, 8, 10, 15, 20, 30, 40, 50]
    dims_ekf = [2, 3, 5, 8, 10, 15, 20, 30, 40, 50]
    dims_mpc = [2, 3, 5, 8, 10, 15, 20]

    print("\n  DARE Solver Scaling:")
    print(f"  {'n':>4}  {'Time (μs)':>12}")
    print("  " + "-" * 18)
    dare_results = benchmark_dare_scaling(dims_dare)
    for n, t in dare_results:
        print(f"  {n:>4}  {t:>12.1f}")
    exp_dare = fit_scaling_exponent(dare_results)
    print(f"  Fitted exponent: O(n^{exp_dare:.2f})")

    print("\n  EKF Predict+Update Scaling:")
    print(f"  {'n':>4}  {'Time (μs)':>12}")
    print("  " + "-" * 18)
    ekf_results = benchmark_ekf_scaling(dims_ekf)
    for n, t in ekf_results:
        print(f"  {n:>4}  {t:>12.1f}")
    exp_ekf = fit_scaling_exponent(ekf_results)
    print(f"  Fitted exponent: O(n^{exp_ekf:.2f})")

    print("\n  MPC (ADMM) Scaling:")
    print(f"  {'n':>4}  {'Time (μs)':>12}")
    print("  " + "-" * 18)
    mpc_results = benchmark_mpc_scaling(dims_mpc)
    for n, t in mpc_results:
        print(f"  {n:>4}  {t:>12.1f}")
    exp_mpc = fit_scaling_exponent(mpc_results)
    print(f"  Fitted exponent: O(n^{exp_mpc:.2f})")

    print("\n  Summary:")
    print(f"  {'Algorithm':>12}  {'Exponent':>10}  {'Expected':>10}")
    print("  " + "-" * 34)
    print(f"  {'DARE':>12}  {exp_dare:>10.2f}  {'~3.0':>10}")
    print(f"  {'EKF':>12}  {exp_ekf:>10.2f}  {'~3.0':>10}")
    print(f"  {'MPC/ADMM':>12}  {exp_mpc:>10.2f}  {'~3.0':>10}")
    print()


if __name__ == "__main__":
    benchmark_scalability()
