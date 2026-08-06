"""
Backend-independent data logging for PyControls simulations.

Records time-series signals (states, inputs, costs, constraints) independent
of any visualization or integration backend.

Usage:
    log = DataLogger()
    log.log_state(x_hat, t=0.01)
    log.log_input(u, t=0.01)
    log.export_npz("run_001.npz")
    log.export_csv("run_001.csv")
"""

import csv
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class DataLogger:
    """Records time-series data from simulation runs."""

    def __init__(self, max_entries: int = 100_000) -> None:
        self._max = max_entries
        self._data: Dict[str, List[Tuple[Optional[float], Any]]] = defaultdict(list)

    def log(self, channel: str, value: Any, t: Optional[float] = None) -> None:
        """Log a value to a named channel."""
        entries = self._data[channel]
        if len(entries) < self._max:
            entries.append((t, value))

    def log_state(self, state: NDArray[Any], t: float) -> None:
        """Log a state vector."""
        self.log("state", state.flatten().tolist(), t)

    def log_input(self, u: Union[float, NDArray[Any]], t: float) -> None:
        """Log a control input."""
        val = u if isinstance(u, (int, float)) else u.flatten().tolist()
        self.log("input", val, t)

    def log_cost(self, cost: float, t: float) -> None:
        """Log an optimization cost value."""
        self.log("cost", cost, t)

    def log_constraint(
        self, violated: bool, details: str, t: float
    ) -> None:
        """Log a constraint evaluation."""
        self.log("constraint", {"violated": violated, "details": details}, t)

    def get(self, channel: str) -> List[Tuple[Optional[float], Any]]:
        """Retrieve all entries from a channel."""
        return self._data.get(channel, [])

    def channels(self) -> List[str]:
        """List all channels that have data."""
        return list(self._data.keys())

    def clear(self) -> None:
        """Clear all logged data."""
        self._data.clear()

    def export_npz(self, path: str, channels: Optional[List[str]] = None) -> None:
        """Export selected channels (or all) as a NumPy .npz archive."""
        targets = channels or list(self._data.keys())
        arrays: Dict[str, NDArray[Any]] = {}
        for ch in targets:
            entries = self._data.get(ch, [])
            if not entries:
                continue
            times = np.array([e[0] for e in entries], dtype=float)
            values = np.array([e[1] for e in entries])
            arrays[f"{ch}_t"] = times
            arrays[f"{ch}_v"] = values
        np.savez(path, **arrays)

    def export_csv(self, path: str, channel: str = "state") -> None:
        """Export a single channel as CSV with timestamp column."""
        entries = self._data.get(channel, [])
        if not entries:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            sample = entries[0][1]
            if isinstance(sample, list):
                header = ["t"] + [f"{channel}_{i}" for i in range(len(sample))]
            else:
                header = ["t", channel]
            writer.writerow(header)
            for t, val in entries:
                if isinstance(val, list):
                    writer.writerow([t] + val)
                else:
                    writer.writerow([t, val])
