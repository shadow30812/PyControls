import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


class TransferFunction:
    """
    Representation of a Single-Input Single-Output (SISO) Transfer Function.
    G(s) = Num(s) / Den(s)
    """

    def __init__(self, num, den):
        self.num = np.array(num, dtype=float)
        self.den = np.array(den, dtype=float)
        self.repr_num = num
        self.repr_den = den

    def __repr__(self):
        """
        String representation of the transfer function.
        """
        return f"TF(Num={self.repr_num}, Den={self.repr_den})"

    def evaluate(self, s):
        """Evaluates G(s) at a complex number s using Horner's method (via np.polyval)."""
        n_val = np.polyval(self.num, s)
        d_val = np.polyval(self.den, s)
        return n_val / d_val if d_val != 0 else np.inf

    def bode_response(self, omega_range):
        """Calculates Magnitude (dB) and Phase (deg) over a frequency range."""
        omega = np.asarray(omega_range)
        mags = np.empty_like(omega, dtype=float)
        phases = np.empty_like(omega, dtype=float)

        for k, w in enumerate(omega):
            s = 1j * w
            resp = self.evaluate(s)
            mags[k] = 20.0 * np.log10(np.abs(resp))
            phases[k] = np.degrees(np.angle(resp))

        return mags, phases

    def to_state_space(self):
        """
        Converts the SISO Transfer Function to State-Space Control Canonical Form.

        Returns:
            tuple: (A, B, C, D) matrices.
        """
        norm = self.den[0]
        a = self.den / norm
        b = self.num / norm

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
