"""
Benchmark 1.4 — DARE Solver Convergence.

Compares the iterative solve_discrete_riccati from core.control_utils against
scipy.linalg.solve_discrete_are for randomly generated stabilizable systems
of varying dimensions (2–20).

Expected result: Iterative convergence with residuals approaching machine
precision, matching scipy's QZ-decomposition accuracy.

Run from the project root:
    python -m benchmarking.benchmark_dare
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import scipy.linalg
except ImportError:
    print("ERROR: scipy is required for this benchmark.")
    print("Install it with:  pip install scipy")
    sys.exit(1)

from core.control_utils import solve_discrete_riccati


def _make_stabilizable_system(n, m=1):
    """
    Generates a random discrete-time system (A, B, Q, R) that is stabilizable,
    ensuring the DARE has a solution.
    """
                                                                      
                                             
    eigvals = np.random.uniform(0.2, 0.95, n) * np.sign(np.random.randn(n))
    V = np.random.randn(n, n)
    while np.abs(np.linalg.det(V)) < 1e-3:
        V = np.random.randn(n, n)
    A = V @ np.diag(eigvals) @ np.linalg.inv(V)

    B = np.random.randn(n, m)
    Q = np.eye(n)                  
    R = np.eye(m) * 0.1                 

    return A, B, Q, R


def benchmark_dare():
    """Runs the DARE convergence benchmark."""
    test_dims = [2, 3, 4, 5, 8, 10, 15, 20]
    results = []

    print("=" * 60)
    print("Benchmark 1.4: DARE Solver Convergence")
    print("=" * 60)
    print(
        f"  {'Dim':>4}  {'Scipy Norm':>12}  {'Custom Norm':>12}  {'Rel Error':>12}  {'Residual':>12}"
    )
    print("  " + "-" * 56)

    for n in test_dims:
        try:
            A, B, Q, R = _make_stabilizable_system(n)

                                                             
            P_scipy = scipy.linalg.solve_discrete_are(A, B, Q, R)

                                       
            P_custom = solve_discrete_riccati(A, B, Q, R, tol=1e-12, max_iter=2000)

                                                      
            rel_err = np.linalg.norm(P_scipy - P_custom) / (
                np.linalg.norm(P_scipy) + 1e-16
            )

                                                                  
            BTP = B.T @ P_custom
            S = R + BTP @ B
            K = np.linalg.solve(S, BTP @ A)
            P_rhs = A.T @ P_custom @ A - A.T @ P_custom @ B @ K + Q
            residual = np.linalg.norm(P_custom - P_rhs)

            print(
                f"  {n:>4}  "
                f"{np.linalg.norm(P_scipy):>12.4e}  "
                f"{np.linalg.norm(P_custom):>12.4e}  "
                f"{rel_err:>12.2e}  "
                f"{residual:>12.2e}"
            )
            results.append((n, rel_err, residual))

        except Exception as e:
            print(f"  {n:>4}  FAILED: {e}")

    print()
    if results:
        max_rel = max(r[1] for r in results)
        max_res = max(r[2] for r in results)
        if max_rel < 1e-6:
            print("  ✓ PASS — Custom DARE matches scipy across all dimensions.")
        else:
            print(f"  ⚠ WARNING — Max relative error: {max_rel:.2e}")
        print(f"  Max residual: {max_res:.2e}")
    print()


if __name__ == "__main__":
    benchmark_dare()
