"""
Benchmark Runner — Executes all profiling and benchmarking scripts.

Mirrors the pattern of test_runner.py in the project root. Runs each
benchmark module sequentially and reports a final summary.

Run from the project root:
    python benchmark_runner.py
"""

import importlib
import sys
import time
import traceback

BENCHMARKS = [
    ("benchmarking.benchmark_csd", "1.1  CSD Accuracy", "benchmark_csd"),
    (
        "benchmarking.benchmark_expm",
        "1.2  Matrix Exponential",
        "benchmark_matrix_exponential",
    ),
    ("benchmarking.benchmark_ode", "1.3  ODE Solver Accuracy", "benchmark_ode"),
    ("benchmarking.benchmark_dare", "1.4  DARE Convergence", "benchmark_dare"),
    ("benchmarking.benchmark_jit", "2.1  JIT Speedup", "benchmark_jit_speedup"),
    ("benchmarking.benchmark_mpc", "2.2  MPC Solve Times", "benchmark_mpc"),
    (
        "benchmarking.benchmark_ekf_ukf_timing",
        "2.3  EKF/UKF Timing",
        "benchmark_ekf_ukf",
    ),
    (
        "benchmarking.benchmark_scalability",
        "2.4  Scalability Study",
        "benchmark_scalability",
    ),
    ("benchmarking.control_metrics", "3.1  Control Metrics", "demo_control_metrics"),
    (
        "benchmarking.benchmark_mpc_optimality",
        "3.3  MPC Optimality",
        "benchmark_mpc_optimality",
    ),
    (
        "benchmarking.benchmark_stability_margins",
        "3.4  Stability Margins",
        "benchmark_stability_margins",
    ),
    (
        "benchmarking.estimation_metrics",
        "4.1–4.3  Estimation Metrics",
        "benchmark_estimation",
    ),
]


def run_all_benchmarks():
    """Discovers and runs all benchmark modules."""
    print("╔" + "═" * 62 + "╗")
    print("║" + "  PyControls — Full Benchmark Suite".center(62) + "║")
    print("╚" + "═" * 62 + "╝")
    print()

    results = []

    for module_path, display_name, func_name in BENCHMARKS:
        print(f"▶ Running: {display_name}")
        print("-" * 64)

        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(module_path)
            entry = getattr(mod, func_name)
            entry()
            elapsed = time.perf_counter() - t0
            results.append((display_name, "PASS", elapsed))
        except Exception as e:
            elapsed = time.perf_counter() - t0
            results.append((display_name, f"FAIL: {e}", elapsed))
            traceback.print_exc()
            print()

    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + "  Benchmark Summary".center(62) + "║")
    print("╚" + "═" * 62 + "╝")
    print()
    print(f"  {'Benchmark':<30} {'Status':<10} {'Time (s)':>10}")
    print("  " + "-" * 52)

    passed = 0
    failed = 0
    total_time = 0.0

    for name, status, elapsed in results:
        total_time += elapsed
        if status == "PASS":
            passed += 1
            status_str = "✓ PASS"
        else:
            failed += 1
            status_str = "✗ FAIL"
        print(f"  {name:<30} {status_str:<10} {elapsed:>10.2f}")

    print("  " + "-" * 52)
    print(f"  {'TOTAL':<30} {passed}P/{failed}F    {total_time:>10.2f}")
    print()

    if failed > 0:
        print(f"  {failed} benchmark(s) failed. Check output above for details.")
    else:
        print("  All benchmarks completed successfully.")
    print()

    return failed


if __name__ == "__main__":
    failures = run_all_benchmarks()
    sys.exit(failures)
