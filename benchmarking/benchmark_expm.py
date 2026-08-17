"""
Benchmark 1.2 — Matrix Exponential Accuracy.

Compares the custom JIT-compiled manual_matrix_exp (scaling & squaring + Taylor)
against scipy.linalg.expm (Padé approximants + scaling & squaring) across 1000
random matrices of varying dimensions and norms.

Expected result: Agreement to approximately 12–14 decimal places.

Run from the project root:
    python -m benchmarking.benchmark_expm
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

from core.solver import manual_matrix_exp


def benchmark_matrix_exponential():
    """Runs the matrix exponential accuracy benchmark."""
    n_tests = 1000
    dims = [2, 3, 4, 5, 10]
    errors = []

    for _ in range(n_tests):
        n = np.random.choice(dims)
                                                             
        scale = 10 ** np.random.uniform(-3, 1.0)
        A = np.random.randn(n, n) * scale

        exp_scipy = scipy.linalg.expm(A)
        exp_custom = manual_matrix_exp(A)

        err = np.linalg.norm(exp_scipy - exp_custom) / (
            np.linalg.norm(exp_scipy) + 1e-16
        )
        errors.append(err)

    errors = np.array(errors)
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    p99_err = np.percentile(errors, 99)
    median_err = np.median(errors)

    print("=" * 60)
    print("Benchmark 1.2: Matrix Exponential Accuracy")
    print("=" * 60)
    print(f"  Tests run:            {n_tests}")
    print(f"  Dimensions tested:    {dims}")
    print(f"  Mean Error:           {mean_err:.2e}")
    print(f"  Median Error:         {median_err:.2e}")
    print(f"  99th Percentile Err:  {p99_err:.2e}")
    print(f"  Max Error:            {max_err:.2e}")
    if max_err < 1e-8:
        print("  ✓ PASS — Custom expm matches scipy reference to high precision.")
    else:
        print("  ⚠ WARNING — Max error exceeds 10⁻⁸. Investigate edge cases.")
    print()


if __name__ == "__main__":
    benchmark_matrix_exponential()
