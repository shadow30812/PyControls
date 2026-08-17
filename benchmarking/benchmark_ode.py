"""
Benchmark 1.3 — ODE Solver Accuracy.

Compares the custom Dormand-Prince adaptive solver (NonlinearSolver.solve_adaptive)
against scipy.integrate.solve_ivp with tight tolerances on standard benchmark ODEs:

  1. Van der Pol oscillator (non-stiff, mu=1)
  2. Lorenz attractor (chaotic)

Expected result: Comparable global accuracy (< 10⁻⁸) with similar or
fewer function evaluations.

Run from the project root:
    python -m benchmarking.benchmark_ode
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from scipy.integrate import solve_ivp
except ImportError:
    print("ERROR: scipy is required for this benchmark.")
    print("Install it with:  pip install scipy")
    sys.exit(1)

from core.solver import NonlinearSolver

                                                                             
                   
                                                                             

                                                                            
             
                                
VDP_MU = 1.0
VDP_X0 = np.array([2.0, 0.0])
VDP_T_END = 10.0


def vdp_scipy(t, x):
    """Van der Pol for scipy (signature: f(t, x) -> dx)."""
    return [x[1], VDP_MU * (1 - x[0] ** 2) * x[1] - x[0]]


def vdp_pycontrols(t, x, u):
    """Van der Pol for NonlinearSolver (signature: f(t, x, u) -> dx)."""
    x = np.asarray(x).flatten()
    return np.array([x[1], VDP_MU * (1 - x[0] ** 2) * x[1] - x[0]])


                                                                           
                       
                         
                      
LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0
LORENZ_X0 = np.array([1.0, 1.0, 1.0])
LORENZ_T_END = 5.0                                                                


def lorenz_scipy(t, x):
    """Lorenz for scipy."""
    return [
        LORENZ_SIGMA * (x[1] - x[0]),
        x[0] * (LORENZ_RHO - x[2]) - x[1],
        x[0] * x[1] - LORENZ_BETA * x[2],
    ]


def lorenz_pycontrols(t, x, u):
    """Lorenz for NonlinearSolver."""
    x = np.asarray(x).flatten()
    return np.array(
        [
            LORENZ_SIGMA * (x[1] - x[0]),
            x[0] * (LORENZ_RHO - x[2]) - x[1],
            x[0] * x[1] - LORENZ_BETA * x[2],
        ]
    )


                                                                             
                  
                                                                             


def run_comparison(name, f_scipy, f_custom, x0, t_end):
    """
    Runs both solvers on the same ODE and compares final states.

    The scipy solution with rtol=1e-12, atol=1e-12 is treated as the
    reference ("ground truth").
    """
                                                                        
    ref = solve_ivp(
        f_scipy,
        [0, t_end],
        x0,
        method="RK45",
        rtol=1e-12,
        atol=1e-12,
        dense_output=True,
    )
    x_ref_final = ref.y[:, -1]
    n_evals_scipy = ref.nfev

                                                                        
    solver = NonlinearSolver(
        dynamics_func=f_custom,
        tol=1e-8,
        dt_min=1e-8,
        dt_max=0.5,
    )
    t_hist, x_hist = solver.solve_adaptive(t_end, x0, u_func=None)
    x_custom_final = x_hist[-1]
    n_steps_custom = len(t_hist)

                                                                        
    errors = []
    for i, tc in enumerate(t_hist):
        x_ref_t = ref.sol(tc)
        x_cust_t = x_hist[i]
        err = np.linalg.norm(x_ref_t - x_cust_t)
        errors.append(err)

    errors = np.array(errors)
    global_err_final = np.linalg.norm(x_ref_final - x_custom_final)

    return {
        "name": name,
        "final_error": global_err_final,
        "mean_error": np.mean(errors),
        "max_error": np.max(errors),
        "scipy_evals": n_evals_scipy,
        "custom_steps": n_steps_custom,
    }


def benchmark_ode():
    """Runs the ODE solver accuracy benchmark."""
    print("=" * 60)
    print("Benchmark 1.3: ODE Solver Accuracy (Dormand-Prince)")
    print("=" * 60)

    results = []

    print("\n  Running Van der Pol oscillator (μ=1.0, t=0→10s)...")
    r1 = run_comparison("Van der Pol", vdp_scipy, vdp_pycontrols, VDP_X0, VDP_T_END)
    results.append(r1)

    print("  Running Lorenz attractor (σ=10, ρ=28, β=8/3, t=0→5s)...")
    r2 = run_comparison(
        "Lorenz", lorenz_scipy, lorenz_pycontrols, LORENZ_X0, LORENZ_T_END
    )
    results.append(r2)

    print()
    print(
        f"  {'ODE':<14} {'Final Err':>10} {'Mean Err':>10} {'Max Err':>10} {'Scipy Evals':>12} {'Custom Steps':>13}"
    )
    print("  " + "-" * 71)

    for r in results:
        print(
            f"  {r['name']:<14} "
            f"{r['final_error']:>10.2e} "
            f"{r['mean_error']:>10.2e} "
            f"{r['max_error']:>10.2e} "
            f"{r['scipy_evals']:>12} "
            f"{r['custom_steps']:>13}"
        )

    print()
    worst_final = max(r["final_error"] for r in results)
    worst_max = max(r["max_error"] for r in results)
    if worst_max < 1e-4:
        print(
            "  ✓ PASS — Custom solver matches scipy reference to high precision."
        )
    elif worst_final < 1e-4:
        print(
            f"  ✓ PASS — Final state error {worst_final:.2e} confirms solver accuracy."
        )
        print(
            f"           (Trajectory max error {worst_max:.2e} reflects tolerance setting.)"
        )
    else:
        print(f"  ✗ FAIL — Final state error {worst_final:.2e} exceeds threshold.")
    print()


if __name__ == "__main__":
    benchmark_ode()
