"""
Benchmark 1.1 — Complex-Step Differentiation (CSD) Accuracy.

Compares CSD-computed Jacobians (via core.math_utils.jacobian) against
analytically derived Jacobians for the DC Motor system.

Expected result: Relative error on the order of machine epsilon (< 10⁻¹⁵).

Run from the project root:
    python -m benchmarking.benchmark_csd
"""

import os
import sys

import numpy as np

                                        
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.math_utils import jacobian
from systems.dc_motor import DCMotor


def analytical_jacobian_dcmotor(state, u, params):
    """
    Returns the exact (analytically derived) A and B Jacobians for the DC Motor.

    State equations:
        d(omega)/dt = (-b/J)*omega + (K/J)*i - (1/J)*T_load
        d(i)/dt     = (-K/L)*omega - (R/L)*i + (1/L)*V

    A = df/dx,  B = df/du  (where u = [V, T_load])
    """
    R = params["R"]
    L = params["L"]
    K = params["K"]
    J = params["J"]
    b = params["b"]

    A = np.array([[-b / J, K / J], [-K / L, -R / L]])
    B = np.array([[0.0, -1.0 / J], [1.0 / L, 0.0]])
    return A, B


def benchmark_csd():
    """Runs the CSD accuracy benchmark over 1000 random test points."""
    system = DCMotor()
    params = system.params

    J = params["J"]
    b = params["b"]
    K = params["K"]
    R = params["R"]
    L = params["L"]

                                                                            
    def motor_dynamics_x(x):
        """dx/dt as a function of state x, with u held constant (closure)."""
        omega, i = x[0], x[1]
        dw = (-b / J) * omega + (K / J) * i
        di = (-K / L) * omega - (R / L) * i
        return np.array([dw, di])

    def motor_dynamics_u(u_val):
        """dx/dt as a function of input u, with x held constant (closure)."""
                             
        dw = -u_val[1] / J
        di = u_val[0] / L
        return np.array([dw, di])

    n_tests = 1000
    max_err_A = 0.0
    max_err_B = 0.0

    for _ in range(n_tests):
        x0 = np.random.randn(2)
        u0 = np.random.randn(2)

                              
        A_exact, B_exact = analytical_jacobian_dcmotor(x0, u0, params)

                                                                        
                                                                            
                                                

                                                         
        def f_state(x, _u0=u0):
            omega, i = x[0], x[1]
            dw = (-b / J) * omega + (K / J) * i + (K / J) * 0 + (-1.0 / J) * _u0[1]
            di = (-K / L) * omega - (R / L) * i + (1.0 / L) * _u0[0]
            return np.array([dw, di])

        def f_input(u, _x0=x0):
            dw = (-b / J) * _x0[0] + (K / J) * _x0[1] + (-1.0 / J) * u[1]
            di = (-K / L) * _x0[0] - (R / L) * _x0[1] + (1.0 / L) * u[0]
            return np.array([dw, di])

        A_csd = jacobian(f_state, x0)
        B_csd = jacobian(f_input, u0)

                         
        err_A = np.linalg.norm(A_exact - A_csd) / (np.linalg.norm(A_exact) + 1e-16)
        err_B = np.linalg.norm(B_exact - B_csd) / (np.linalg.norm(B_exact) + 1e-16)
        max_err_A = max(max_err_A, err_A)
        max_err_B = max(max_err_B, err_B)

    print("=" * 60)
    print("Benchmark 1.1: Complex-Step Differentiation Accuracy")
    print("=" * 60)
    print(f"  Tests run:                  {n_tests}")
    print(f"  CSD Jacobian A Max Rel Err: {max_err_A:.2e}")
    print(f"  CSD Jacobian B Max Rel Err: {max_err_B:.2e}")
    print(f"  Machine epsilon (float64):  {np.finfo(float).eps:.2e}")
    if max_err_A < 1e-12 and max_err_B < 1e-12:
        print("  ✓ PASS — Machine-precision accuracy confirmed.")
    else:
        print("  ✗ FAIL — Errors exceed expected threshold.")
    print()


if __name__ == "__main__":
    benchmark_csd()
