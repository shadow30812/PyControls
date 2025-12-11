import numpy as np

from core.math_utils import find_root


def get_stability_margins(tf, w_guess=10.0):
    """Finds Margins using Newton-Raphson."""

    def phase_eq_minus_180(w):
        if w <= 0:
            return 180
        s = 1j * w
        resp = tf.evaluate(s)
        phase = np.degrees(np.angle(resp))
        while phase > 0:
            phase -= 360
        return phase + 180

    def mag_eq_0db(w):
        if w <= 0:
            return -100
        s = 1j * w
        resp = tf.evaluate(s)
        return 20 * np.log10(np.abs(resp))

    # Gain Margin (Phase Crossover)
    w_pc = find_root(phase_eq_minus_180, w_guess)
    gain_margin = float("inf")
    if w_pc > 0 and w_pc < 1e5:
        mag_at_pc = mag_eq_0db(w_pc)
        gain_margin = -mag_at_pc

    # Phase Margin (Gain Crossover)
    w_gc = find_root(mag_eq_0db, w_guess)
    phase_margin = float("inf")
    if w_gc > 0 and w_gc < 1e5:
        s = 1j * w_gc
        resp = tf.evaluate(s)
        phase_at_gc = np.degrees(np.angle(resp))
        while phase_at_gc > 0:
            phase_at_gc -= 360
        phase_margin = 180 + phase_at_gc

    return gain_margin, phase_margin, w_pc, w_gc


def get_step_metrics(time, response):
    final_val = response[-1]
    if final_val == 0:
        return 0, 0, 0

    try:
        t_10_idx = np.where(response >= 0.1 * final_val)[0][0]
        t_90_idx = np.where(response >= 0.9 * final_val)[0][0]
        rise_time = time[t_90_idx] - time[t_10_idx]
    except IndexError:
        rise_time = 0

    peak_val = np.max(response)
    overshoot = ((peak_val - final_val) / final_val) * 100

    upper = final_val * 1.02
    lower = final_val * 0.98
    mask = (response > upper) | (response < lower)

    settling_time = time[np.where(mask)[0][-1]] if np.any(mask) else 0

    return rise_time, overshoot, settling_time
