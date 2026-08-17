"""
Benchmark 3.3 — MPC Optimality vs Horizon Length.

Sweeps MPC prediction horizon H from small to large, computes total trajectory
cost, and compares to the theoretical LQR infinite-horizon cost.

Expected result: MPC cost converges asymptotically to LQR optimal cost
as horizon increases.

Run from the project root:
    python -m benchmarking.benchmark_mpc_optimality
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.control_utils import dlqr
from core.mpc import ModelPredictiveControl
from systems.dc_motor import DCMotor


def compute_lqr_cost(A_d, B_d, Q, R, x0, x_ref, K_lqr, n_sim_steps):
    """
    Simulates LQR control and computes the total trajectory cost.

    Cost = Σ (x-xref)'Q(x-xref) + u'Ru
    """
    x = x0.copy().reshape(-1)
    x_ref_flat = x_ref.reshape(-1)
    total_cost = 0.0

    for _ in range(n_sim_steps):
        e = x - x_ref_flat
        u = -K_lqr @ e
        total_cost += float(e @ Q @ e + u @ R @ u)
        x = (A_d @ x.reshape(-1, 1) + B_d @ u.reshape(-1, 1)).flatten()

    return total_cost


def compute_mpc_cost(A_d, B_d, Q, R, x0, x_ref, horizon, dt, n_sim_steps):
    """
    Simulates MPC control and computes the total trajectory cost.
    """
    mpc = ModelPredictiveControl(
        A=A_d,
        B=B_d,
        horizon=horizon,
        dt=dt,
        Q=Q,
        R=R,
        u_min=-100.0,
        u_max=100.0,
    )

    x = x0.copy().reshape(-1)
    x_ref_flat = x_ref.reshape(-1)
    total_cost = 0.0

    for _ in range(n_sim_steps):
        e = x - x_ref_flat
        u = mpc.optimize(x, x_ref_flat)
        total_cost += float(e @ Q @ e + u @ R @ u)
        x = (A_d @ x.reshape(-1, 1) + B_d @ u.reshape(-1, 1)).flatten()

    return total_cost


def benchmark_mpc_optimality():
    """Runs the MPC optimality vs horizon length benchmark."""
    motor = DCMotor()
    dt = 0.05
    A_d, B_d = motor.get_mpc_model(dt)

    Q = np.diag([10.0, 1.0])
    R = np.diag([1.0])

    x0 = np.array([10.0, 5.0])
    x_ref = np.array([0.0, 0.0])
    n_sim_steps = 60

    K_lqr = dlqr(A_d, B_d, Q, R)
    lqr_cost = compute_lqr_cost(A_d, B_d, Q, R, x0, x_ref, K_lqr, n_sim_steps)

    horizons = [2, 3, 5, 8, 10, 15, 20]

    print("=" * 60)
    print("Benchmark 3.3: MPC Optimality vs Horizon Length")
    print("=" * 60)
    print(f"\n  LQR Optimal Cost (infinite horizon): {lqr_cost:.4f}")
    print(f"  Simulation steps: {n_sim_steps}, dt={dt}s")
    print()
    print(f"  {'Horizon':>8}  {'MPC Cost':>12}  {'% of LQR':>10}  {'Gap (%)':>10}")
    print("  " + "-" * 44)

    for H in horizons:
        mpc_cost = compute_mpc_cost(A_d, B_d, Q, R, x0, x_ref, H, dt, n_sim_steps)
        pct_of_lqr = (mpc_cost / lqr_cost) * 100 if lqr_cost != 0 else float("inf")
        gap = pct_of_lqr - 100.0
        print(
            f"  {H:>8}  "
            f"{mpc_cost:>12.4f}  "
            f"{pct_of_lqr:>10.2f}  "
            f"{gap:>+10.2f}"
        )

    print()
    print("  Expected: MPC cost → LQR cost as H → ∞")
    print()


if __name__ == "__main__":
    benchmark_mpc_optimality()
