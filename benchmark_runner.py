"""
Benchmark Runner — Executes all profiling and benchmarking scripts.

Runs benchmark modules across multiple iterations (default 5 runs),
computes statistical metrics (mean, min, max, std dev), and saves raw
and aggregated benchmark results to files in the benchmarking/ folder.

Run from the project root:
    python benchmark_runner.py
    python benchmark_runner.py --runs 5
"""

import argparse
import datetime
import importlib
import json
import os
import platform
import sys
import time
import traceback

import numpy as np

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
    (
        "benchmarking.benchmark_scipy_speed",
        "5.1  SciPy Speed Comparison",
        "benchmark_scipy_speed",
    ),
]


def run_benchmark_suite(num_runs: int = 5):
    """
    Executes all benchmark modules across num_runs iterations and computes statistics.
    """
    print("╔" + "═" * 70 + "╗")
    print("║" + f"  PyControls — Benchmark Suite ({num_runs} Iterations Average)".center(70) + "║")
    print("╚" + "═" * 70 + "╝")
    print()

    run_records = []
    benchmark_timings = {name: [] for _, name, _ in BENCHMARKS}
    benchmark_statuses = {name: [] for _, name, _ in BENCHMARKS}

    total_suite_start = time.perf_counter()

    for run_idx in range(1, num_runs + 1):
        print(f"\n{'━' * 72}")
        print(f"  Iteration {run_idx} of {num_runs}")
        print(f"{'━' * 72}\n")

        current_run_results = {}

        for module_path, display_name, func_name in BENCHMARKS:
            print(f"▶ [{run_idx}/{num_runs}] Running: {display_name}")
            print("-" * 64)

            t0 = time.perf_counter()
            status = "PASS"
            err_msg = ""
            try:
                mod = importlib.import_module(module_path)
                entry = getattr(mod, func_name)
                entry()
                elapsed = time.perf_counter() - t0
            except Exception as e:
                elapsed = time.perf_counter() - t0
                status = "FAIL"
                err_msg = str(e)
                traceback.print_exc()
                print()

            benchmark_timings[display_name].append(elapsed)
            benchmark_statuses[display_name].append(status)
            current_run_results[display_name] = {
                "status": status,
                "elapsed_sec": elapsed,
                "error": err_msg,
            }

        run_records.append({
            "run_index": run_idx,
            "timestamp": datetime.datetime.now().isoformat(),
            "benchmarks": current_run_results,
        })

    total_suite_elapsed = time.perf_counter() - total_suite_start

    summary_stats = []
    total_mean_time = 0.0
    total_failures = 0

    for _, display_name, _ in BENCHMARKS:
        times = np.array(benchmark_timings[display_name])
        statuses = benchmark_statuses[display_name]

        pass_count = statuses.count("PASS")
        fail_count = statuses.count("FAIL")
        total_failures += fail_count

        mean_t = float(np.mean(times))
        min_t = float(np.min(times))
        max_t = float(np.max(times))
        std_t = float(np.std(times))

        total_mean_time += mean_t

        summary_stats.append({
            "name": display_name,
            "mean_sec": mean_t,
            "min_sec": min_t,
            "max_sec": max_t,
            "std_sec": std_t,
            "passes": pass_count,
            "fails": fail_count,
            "all_runs_sec": [float(t) for t in times],
        })

    print("\n")
    print("╔" + "═" * 72 + "╗")
    print("║" + f"  PyControls — Aggregate Benchmark Summary ({num_runs} Runs)".center(72) + "║")
    print("╚" + "═" * 72 + "╝")
    print()

    header = f"  {'Benchmark':<28} {'Status':<9} {'Mean (s)':>9} {'Min (s)':>9} {'Max (s)':>9} {'StdDev':>9}"
    print(header)
    print("  " + "─" * 74)

    for stat in summary_stats:
        status_str = f"{stat['passes']}/{num_runs} P" if stat["fails"] == 0 else f"{stat['fails']} FAIL"
        print(
            f"  {stat['name']:<28} "
            f"{status_str:<9} "
            f"{stat['mean_sec']:>9.2f} "
            f"{stat['min_sec']:>9.2f} "
            f"{stat['max_sec']:>9.2f} "
            f"{stat['std_sec']:>9.2f}"
        )

    print("  " + "─" * 74)
    overall_status = f"{len(BENCHMARKS) * num_runs - total_failures}/{len(BENCHMARKS) * num_runs} PASS"
    print(f"  {'TOTAL / AVERAGE':<28} {overall_status:<9} {total_mean_time:>9.2f} s")
    print(f"  Wall-clock suite time: {total_suite_elapsed:.2f} s")
    print()

    results_data = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "num_runs": num_runs,
            "total_wall_clock_sec": total_suite_elapsed,
            "total_mean_sec": total_mean_time,
            "total_failures": total_failures,
        },
        "summary": summary_stats,
        "raw_runs": run_records,
    }

    benchmarks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarking")
    json_path = os.path.join(benchmarks_dir, "benchmark_results.json")
    txt_path = os.path.join(benchmarks_dir, "benchmark_results.txt")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"PyControls Benchmark Results — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Platform: {platform.platform()}\n")
            f.write(f"Iterations: {num_runs} | Total Wall-Clock: {total_suite_elapsed:.2f}s\n\n")
            f.write(header + "\n")
            f.write("  " + "─" * 74 + "\n")
            for stat in summary_stats:
                status_str = f"{stat['passes']}/{num_runs} P" if stat["fails"] == 0 else f"{stat['fails']} FAIL"
                f.write(
                    f"  {stat['name']:<28} "
                    f"{status_str:<9} "
                    f"{stat['mean_sec']:>9.2f} "
                    f"{stat['min_sec']:>9.2f} "
                    f"{stat['max_sec']:>9.2f} "
                    f"{stat['std_sec']:>9.2f}\n"
                )
            f.write("  " + "─" * 74 + "\n")
            f.write(f"  {'TOTAL / AVERAGE':<28} {overall_status:<9} {total_mean_time:>9.2f} s\n\n")

            f.write("\nDetailed Raw Iteration Times (seconds):\n")
            f.write(f"{'Benchmark':<30} " + " ".join([f"Run {i+1:>6}" for i in range(num_runs)]) + "\n")
            f.write("-" * (30 + 8 * num_runs) + "\n")
            for stat in summary_stats:
                runs_str = " ".join([f"{t:>8.2f}" for t in stat["all_runs_sec"]])
                f.write(f"{stat['name']:<30} {runs_str}\n")

        print(f"  ✓ Raw results saved to: {os.path.relpath(json_path)}")
        print(f"  ✓ Text summary saved to: {os.path.relpath(txt_path)}")
    except Exception as e:
        print(f"  Warning: Failed to save results to file: {e}")

    print()
    return total_failures


def main():
    """CLI entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="PyControls Benchmark Suite Runner")
    parser.add_argument(
        "runs_pos",
        nargs="?",
        type=int,
        default=None,
        help="Optional positional number of benchmark iterations (default: 5)",
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=5,
        help="Number of iterations to run and average (default: 5)",
    )
    args = parser.parse_args()

    num_runs = args.runs_pos if args.runs_pos is not None else args.runs
    if num_runs < 1:
        print("Error: Number of runs must be >= 1.")
        sys.exit(1)

    failures = run_benchmark_suite(num_runs=num_runs)
    sys.exit(failures)


if __name__ == "__main__":
    main()
