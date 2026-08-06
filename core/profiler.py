"""
Lightweight profiler for real-time control loops.

Usage:
    prof = StepProfiler()

    with prof.measure("ekf_predict"):
        ekf.predict(u, dt)

    with prof.measure("mpc_optimize"):
        u = mpc.optimize(x, x_ref)

    print(prof.summary())
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Generator, List


class StepProfiler:
    """Measures per-step timing of control loop components."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._timings: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def measure(self, label: str) -> Generator[None, None, None]:
        """Context manager that records elapsed time in microseconds."""
        if not self._enabled:
            yield
            return
        t0 = time.perf_counter_ns()
        yield
        elapsed_us = (time.perf_counter_ns() - t0) / 1_000.0
        self._timings[label].append(elapsed_us)

    def record(self, label: str, duration_us: float) -> None:
        """Manually record a timing measurement."""
        if self._enabled:
            self._timings[label].append(duration_us)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return statistics for each label: count, mean, min, max, last (all in us)."""
        result: Dict[str, Dict[str, float]] = {}
        for label, times in self._timings.items():
            if not times:
                continue
            result[label] = {
                "count": len(times),
                "mean_us": sum(times) / len(times),
                "min_us": min(times),
                "max_us": max(times),
                "last_us": times[-1],
            }
        return result

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._timings.clear()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
