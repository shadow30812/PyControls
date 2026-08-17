"""
Benchmark 2.1 — Numba JIT Speedup.

Measures the execution time of manual_matrix_exp with JIT compilation
(as-is from core.solver) and compares it against a pure-Python re-implementation
without the @njit decorator.

Expected result: 10–100x speedup from JIT compilation.

Run from the project root:
    python -m benchmarking.benchmark_jit
"""

import os
import sys
import timeit

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.solver import manual_matrix_exp as expm_jit


                                                                             
def _mat_mul_nojit(A, B):
    """Manual matrix multiplication — pure Python, no JIT."""
    rows_A = A.shape[0]
    cols_A = A.shape[1]
    cols_B = B.shape[1]
    C = np.zeros((rows_A, cols_B), dtype=np.float64)
    for i in range(rows_A):
        for j in range(cols_B):
            acc = 0.0
            for k in range(cols_A):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc
    return C


def manual_matrix_exp_nojit(A, order=20):
    """
    Pure-Python matrix exponential (scaling & squaring + Taylor).
    Identical algorithm to core.solver.manual_matrix_exp but without Numba.
    """
    rows, cols = A.shape
    if rows == 1:
        return np.array([[np.exp(A[0, 0])]])

    norm_A = 0.0
    for i in range(rows):
        row_sum = 0.0
        for j in range(cols):
            row_sum += np.abs(A[i, j])
        if row_sum > norm_A:
            norm_A = row_sum

    s = 0
    while norm_A > 0.5:
        norm_A *= 0.5
        s += 1

    inv_scale = 1.0 / (2.0**s)
    A_scaled = A * inv_scale

    E = np.eye(rows)
    term = np.eye(rows)

    for k in range(1, order + 1):
        term = _mat_mul_nojit(term, A_scaled) / k
        E += term

    for _ in range(s):
        E = _mat_mul_nojit(E, E)

    return E


def benchmark_jit_speedup():
    """Runs the JIT speedup benchmark."""
    A = np.random.randn(4, 4) * 0.1

                                                               
    _ = expm_jit(A)

    n_runs_jit = 10000
    n_runs_nojit = 100                                

                          
    time_jit = timeit.timeit(lambda: expm_jit(A), number=n_runs_jit)
    us_per_call_jit = (time_jit / n_runs_jit) * 1e6

                                  
    time_nojit = timeit.timeit(lambda: manual_matrix_exp_nojit(A), number=n_runs_nojit)
    us_per_call_nojit = (time_nojit / n_runs_nojit) * 1e6

    speedup = us_per_call_nojit / us_per_call_jit

    print("=" * 60)
    print("Benchmark 2.1: Numba JIT Compilation Speedup")
    print("=" * 60)
    print(f"  Matrix size:       4×4")
    print(
        f"  No JIT:            {us_per_call_nojit:.2f} μs/call  ({n_runs_nojit} runs)"
    )
    print(f"  With JIT:          {us_per_call_jit:.2f} μs/call  ({n_runs_jit} runs)")
    print(f"  Speedup:           {speedup:.1f}x")
    print()

                             
    A8 = np.random.randn(8, 8) * 0.1
    _ = expm_jit(A8)          

    time_jit_8 = timeit.timeit(lambda: expm_jit(A8), number=n_runs_jit)
    us_jit_8 = (time_jit_8 / n_runs_jit) * 1e6

    time_nojit_8 = timeit.timeit(
        lambda: manual_matrix_exp_nojit(A8), number=n_runs_nojit
    )
    us_nojit_8 = (time_nojit_8 / n_runs_nojit) * 1e6

    speedup_8 = us_nojit_8 / us_jit_8

    print(f"  Matrix size:       8×8")
    print(f"  No JIT:            {us_nojit_8:.2f} μs/call")
    print(f"  With JIT:          {us_jit_8:.2f} μs/call")
    print(f"  Speedup:           {speedup_8:.1f}x")
    print()


if __name__ == "__main__":
    benchmark_jit_speedup()
