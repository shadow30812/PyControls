from math import inf

import numpy as np

from core.math_utils import Root

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


def get_stability_margins(tf, w_start=-2, w_end=5):
    """
    Calculates Gain Margin and Phase Margin of a Transfer Function.

    Methods:
    - Gain Margin: Magnitude at the frequency where Phase = -180 deg.
    - Phase Margin: Phase at the frequency where Magnitude = 0 dB.

    Args:
        tf: TransferFunction object.
        w_start, w_end: Log10 bounds for frequency search.

    Returns:
        tuple: (Gain Margin, Phase Margin, Phase Crossover Freq, Gain Crossover Freq)
    """

    w = np.logspace(w_start, w_end, 200)
    s = 1j * w
    resp = np.array([tf.evaluate(si) for si in s])

    mag_db = 20 * np.log10(np.abs(resp))
    phase = np.angle(resp, deg=True)
    phase = (phase + 180) % 360 - 180

    w_pc = 0.0
    phase_shifted = phase + 180
    sign_change = np.where(
        np.signbit(phase_shifted[:-1]) != np.signbit(phase_shifted[1:])
    )[0]

    if sign_change.size > 0:
        i = sign_change[0]
        try:
            w_pc = Root().find_root(
                lambda x: np.angle(tf.evaluate(1j * x), deg=True) + 180,
                w[i],
                w[i + 1],
            )
        except Exception:
            w_pc = 0.0

    gain_margin = inf
    if w_pc > 0:
        gain_margin = -20 * np.log10(abs(tf.evaluate(1j * w_pc)))

    w_gc = 0.0
    sign_change = np.where(np.signbit(mag_db[:-1]) != np.signbit(mag_db[1:]))[0]

    if sign_change.size > 0:
        i = sign_change[0]
        try:
            w_gc = Root().find_root(
                lambda x: 20 * np.log10(abs(tf.evaluate(1j * x))),
                w[i],
                w[i + 1],
            )
        except Exception:
            w_gc = 0.0

    phase_margin = inf
    if w_gc > 0:
        phase_at_gc = np.angle(tf.evaluate(1j * w_gc), deg=True)
        phase_at_gc = (phase_at_gc + 180) % 360 - 180
        phase_margin = 180 + phase_at_gc

    return gain_margin, phase_margin, w_pc, w_gc


@njit
def get_exact_time_idx(time, response, target_val):
    """
    Finds the exact time t where response[t] crosses target_val using linear interpolation.
    """
    for i in range(len(response) - 1):
        y1, y2 = response[i], response[i + 1]
        if (y1 <= target_val <= y2) or (y2 <= target_val <= y1):
            t1, t2 = time[i], time[i + 1]

            if y1 == y2:
                return t1
            return t1 + (target_val - y1) * (t2 - t1) / (y2 - y1)

    return 0.0


def get_step_metrics(time, response):
    """
    Computes standard step response metrics: Rise Time, Overshoot, Settling Time.
    """
    final_val = response[-1]
    if final_val == 0:
        return 0, 0, 0

    t_10 = get_exact_time_idx(time, response, 0.1 * final_val)
    t_90 = get_exact_time_idx(time, response, 0.9 * final_val)
    rise_time = max(0.0, t_90 - t_10)

    peak_val = np.max(response)
    overshoot = max(0.0, (peak_val - final_val) * 100 / final_val)

    upper = final_val * 1.02
    lower = final_val * 0.98

    settling_time = 0.0
    for i in range(len(response) - 1, -1, -1):
        if response[i] > upper or response[i] < lower:
            if i + 1 < len(time):
                settling_time = time[i + 1]
            break

    return rise_time, overshoot, settling_time
