import numpy as np


class TransferFunction:
    """
    Representation of a Single-Input Single-Output (SISO) Transfer Function.
    G(s) = Num(s) / Den(s)
    """

    def __init__(self, num, den):
        self.num = np.array(num, dtype=float)
        self.den = np.array(den, dtype=float)

    def evaluate(self, s):
        """Evaluates G(s) at a complex number s using Horner's method (via np.polyval)."""
        n_val = np.polyval(self.num, s)
        d_val = np.polyval(self.den, s)
        return n_val / d_val if d_val != 0 else np.inf

    def bode_response(self, omega_range):
        """Calculates Magnitude (dB) and Phase (deg) over a frequency range."""
        mags = []
        phases = []
        for w in omega_range:
            s = 1j * w
            resp = self.evaluate(s)
            mags.append(20 * np.log10(np.abs(resp)))
            phases.append(np.degrees(np.angle(resp)))
        return np.array(mags), np.array(phases)

    def to_state_space(self):
        """
        Converts the SISO Transfer Function to State-Space Control Canonical Form.

        Returns:
            tuple: (A, B, C, D) matrices.
        """
        norm_factor = self.den[0]
        a = self.den / norm_factor
        b = self.num / norm_factor

        n = len(a) - 1
        if len(b) < len(a):
            b = np.pad(b, (len(a) - len(b), 0), "constant")

        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i + 1] = 1
        A[n - 1, :] = -a[1:][::-1]

        B = np.zeros((n, 1))
        B[n - 1, 0] = 1

        C = (b[1:][::-1] - b[0] * a[1:][::-1]).reshape(1, n)
        D = b[0]

        return A, B, C, D
