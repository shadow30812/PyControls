from math import inf

import numpy as np
from numba import jit

from core.math_utils import Root


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

    def phase_func(w):
        if w <= 0:
            return 180
        s = 1j * w
        resp = tf.evaluate(s)
        phase = np.degrees(np.angle(resp))
        while phase > 0:
            phase -= 360
        return phase + 180

    def mag_func_db(w):
        if w <= 0:
            return -100
        s = 1j * w
        resp = tf.evaluate(s)
        return 20 * np.log10(np.abs(resp))

    w_search = np.logspace(w_start, w_end, 100)

    w_pc = 0.0
    for i in range(len(w_search) - 1):
        wa, wb = w_search[i], w_search[i + 1]
        if phase_func(wa) * phase_func(wb) < 0:
            try:
                w_pc = Root().find_root(phase_func, wa, wb)
                break
            except ValueError:
                continue

    gain_margin = inf
    if w_pc > 0:
        gain_margin = -mag_func_db(w_pc)

    w_gc = 0.0
    for i in range(len(w_search) - 1):
        wa, wb = w_search[i], w_search[i + 1]
        if mag_func_db(wa) * mag_func_db(wb) < 0:
            try:
                w_gc = Root().find_root(mag_func_db, wa, wb)
                break
            except ValueError:
                continue

    phase_margin = inf
    if w_gc > 0:
        s = 1j * w_gc
        resp = tf.evaluate(s)
        phase_at_gc = np.degrees(np.angle(resp))
        while phase_at_gc > 0:
            phase_at_gc -= 360
        phase_margin = 180 + phase_at_gc

    return gain_margin, phase_margin, w_pc, w_gc


@jit(nopython=True)
def get_exact_time_idx(time, response, target_val):
    """
    Finds the exact time t where response[t] crosses target_val using linear interpolation.
    """
    for i in range(len(response) - 1):
        y1 = response[i]
        y2 = response[i + 1]

        if (y1 <= target_val <= y2) or (y1 >= target_val >= y2):
            t1 = time[i]
            t2 = time[i + 1]
            if y2 == y1:
                return t1
            fraction = (target_val - y1) / (y2 - y1)
            return t1 + fraction * (t2 - t1)
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
    rise_time = t_90 - t_10

    peak_val = np.max(response)
    overshoot = ((peak_val - final_val) / final_val) * 100

    upper = final_val * 1.02
    lower = final_val * 0.98

    settling_time = 0
    for i in range(len(response) - 1, 0, -1):
        if response[i] > upper or response[i] < lower:
            settling_time = time[i + 1] if i + 1 < len(time) else time[i]
            break

    return rise_time, overshoot, settling_time
