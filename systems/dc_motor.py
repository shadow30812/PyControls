import numpy as np

from core.transfer_function import TransferFunction


class DCMotor:
    def __init__(self, J=0.01, b=0.1, K=0.01, R=1, L=0.5):
        self.params = {"J": J, "b": b, "K": K, "R": R, "L": L}

    def get_closed_loop_tf(self, Kp, Ki, Kd):
        """Returns T(s) = Ref -> Speed"""
        J, b, K, R, L = self.params.values()

        # Plant G(s)
        p_num = [K]
        p_den = [J * L, (J * R + b * L), (b * R + K**2)]

        # Controller C(s)
        c_num = [Kd, Kp, Ki]
        c_den = [1, 0]

        # T(s) construction
        num = np.convolve(c_num, p_num)
        term1 = np.convolve(c_den, p_den)
        term2 = np.convolve(c_num, p_num)

        # Pad to add
        diff = len(term1) - len(term2)
        if diff > 0:
            term2 = np.pad(term2, (diff, 0), "constant")
        elif diff < 0:
            term1 = np.pad(term1, (-diff, 0), "constant")

        den = term1 + term2
        return TransferFunction(num, den)

    def get_disturbance_tf(self, Kp, Ki, Kd):
        """Returns T_d(s) = Load Torque -> Speed"""
        J, b, K, R, L = self.params.values()

        # Plant Denominator
        p_den = [J * L, (J * R + b * L), (b * R + K**2)]
        # Plant Numerator
        p_num = [K]

        # Controller
        c_num = [Kd, Kp, Ki]
        c_den = [1, 0]

        # Load G(s) = -(Ls+R) / Delta
        g_load_num = [-L, -R]

        # T_d(s) = (g_load_num * c_den) / (p_den * c_den + p_num * c_num)
        num = np.convolve(g_load_num, c_den)

        term1 = np.convolve(p_den, c_den)
        term2 = np.convolve(p_num, c_num)

        diff = len(term1) - len(term2)
        if diff > 0:
            term2 = np.pad(term2, (diff, 0), "constant")
        elif diff < 0:
            term1 = np.pad(term1, (-diff, 0), "constant")

        den = term1 + term2
        return TransferFunction(num, den)
