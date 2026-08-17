"""
Benchmark 3.4 — Stability Margins.

Computes Gain Margin (GM) and Phase Margin (PM) for:
  - DC Motor with PID controller (open-loop transfer function)
  - Inverted Pendulum with LQR controller (loop transfer function)

Expected result: Tabulated stability margins indicating robustness.
(>6dB GM, >45° PM for well-tuned controllers.)

Run from the project root:
    python -m benchmarking.benchmark_stability_margins
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.transfer_function import TransferFunction
from helpers.config import CONTROLLERS
from systems.dc_motor import DCMotor
from systems.pendulum import InvertedPendulum


def compute_gain_phase_margin(tf_obj, omega_range):
    """
    Computes Gain Margin and Phase Margin from a transfer function.

    Gain Margin: |L(jω)| at the frequency where ∠L(jω) = -180°
    Phase Margin: ∠L(jω) + 180° at the frequency where |L(jω)| = 1 (0 dB)

    Args:
        tf_obj: A transfer function object with an `evaluate(s)` method.
        omega_range: Array of frequencies to sweep (rad/s).

    Returns:
        dict: {'GM_dB', 'PM_deg', 'GM_freq', 'PM_freq'}
    """
    mags = []
    phases = []

    for w in omega_range:
        s = 1j * w
        try:
            val = tf_obj.evaluate(s)
            mag = np.abs(val)
            phase = np.degrees(np.angle(val))
            mags.append(mag)
            phases.append(phase)
        except Exception:
            mags.append(np.nan)
            phases.append(np.nan)

    mags = np.array(mags)
    phases = np.array(phases)

                                                                       
    mag_db = 20 * np.log10(mags + 1e-30)
    pm_freq = np.nan
    pm_deg = np.nan

                                       
    for i in range(len(mag_db) - 1):
        if not (np.isnan(mag_db[i]) or np.isnan(mag_db[i + 1])):
            if (mag_db[i] >= 0 and mag_db[i + 1] < 0) or (
                mag_db[i] < 0 and mag_db[i + 1] >= 0
            ):
                                      
                frac = -mag_db[i] / (mag_db[i + 1] - mag_db[i])
                pm_freq = omega_range[i] + frac * (omega_range[i + 1] - omega_range[i])
                phase_at_gc = phases[i] + frac * (phases[i + 1] - phases[i])
                pm_deg = 180.0 + phase_at_gc
                break

                                                              
    gm_freq = np.nan
    gm_db = np.nan

                                       
    phases_unwrapped = np.unwrap(np.radians(phases))
    phases_deg_unwrapped = np.degrees(phases_unwrapped)

    for i in range(len(phases_deg_unwrapped) - 1):
        if not (
            np.isnan(phases_deg_unwrapped[i]) or np.isnan(phases_deg_unwrapped[i + 1])
        ):
            if (
                phases_deg_unwrapped[i] >= -180 and phases_deg_unwrapped[i + 1] < -180
            ) or (
                phases_deg_unwrapped[i] < -180 and phases_deg_unwrapped[i + 1] >= -180
            ):
                frac = (-180.0 - phases_deg_unwrapped[i]) / (
                    phases_deg_unwrapped[i + 1] - phases_deg_unwrapped[i]
                )
                gm_freq = omega_range[i] + frac * (omega_range[i + 1] - omega_range[i])
                mag_at_pc = mag_db[i] + frac * (mag_db[i + 1] - mag_db[i])
                gm_db = -mag_at_pc                                              
                break

    return {
        "GM_dB": gm_db,
        "PM_deg": pm_deg,
        "GM_freq": gm_freq,
        "PM_freq": pm_freq,
    }


def benchmark_stability_margins():
    """Runs the stability margins benchmark."""
    print("=" * 60)
    print("Benchmark 3.4: Stability Margins")
    print("=" * 60)

    omega = np.logspace(-1, 3, 5000)

                                                                
    print("\n  --- DC Motor PID Controllers ---")
    print(
        f"  {'Controller':<18} {'GM (dB)':>8} {'PM (°)':>8} {'GM freq':>10} {'PM freq':>10}"
    )
    print("  " + "-" * 58)

    motor = DCMotor()

    for ctrl in CONTROLLERS:
        tf_open = motor.get_open_loop_tf(ctrl["Kp"], ctrl["Ki"], ctrl["Kd"])
        margins = compute_gain_phase_margin(tf_open, omega)
        gm_str = f"{margins['GM_dB']:.1f}" if not np.isnan(margins["GM_dB"]) else "∞"
        pm_str = (
            f"{margins['PM_deg']:.1f}" if not np.isnan(margins["PM_deg"]) else "N/A"
        )
        gm_f = (
            f"{margins['GM_freq']:.1f}" if not np.isnan(margins["GM_freq"]) else "N/A"
        )
        pm_f = (
            f"{margins['PM_freq']:.1f}" if not np.isnan(margins["PM_freq"]) else "N/A"
        )

        print(f"  {ctrl['name']:<18} {gm_str:>8} {pm_str:>8} {gm_f:>10} {pm_f:>10}")

                                                                
    print("\n  --- Inverted Pendulum LQR ---")
    print(
        f"  {'Config':<18} {'GM (dB)':>8} {'PM (°)':>8} {'GM freq':>10} {'PM freq':>10}"
    )
    print("  " + "-" * 58)

    pend = InvertedPendulum()
    K_lqr = pend.dlqr_gain(dt=0.01)
    lqr_tf = pend.get_open_loop_tf(K_lqr)

    margins = compute_gain_phase_margin(lqr_tf, omega)
    gm_str = f"{margins['GM_dB']:.1f}" if not np.isnan(margins["GM_dB"]) else "∞"
    pm_str = f"{margins['PM_deg']:.1f}" if not np.isnan(margins["PM_deg"]) else "N/A"
    gm_f = f"{margins['GM_freq']:.1f}" if not np.isnan(margins["GM_freq"]) else "N/A"
    pm_f = f"{margins['PM_freq']:.1f}" if not np.isnan(margins["PM_freq"]) else "N/A"

    print(f"  {'LQR (default)':<18} {gm_str:>8} {pm_str:>8} {gm_f:>10} {pm_f:>10}")

    print()
    print("  Target: GM > 6 dB, PM > 45° for robust stability.")
    print()


if __name__ == "__main__":
    benchmark_stability_margins()
