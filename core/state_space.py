import numpy as np

from core.exceptions import DimensionMismatchError


class StateSpace:
    """
    A container for MIMO Linear Time-Invariant (LTI) Systems.
    Defined by the state-space equations:
        dx/dt = Ax + Bu
        y     = Cx + Du
    """

    def __init__(self, A, B, C, D):
        """
        Validates dimensions and stores system matrices.
        Raises DimensionMismatchError if shapes are invalid.
        """
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.C = np.array(C, dtype=float)
        self.D = np.array(D, dtype=float)

        self.n_states = self.A.shape[0]
        self.n_inputs = self.B.shape[1]
        self.n_outputs = self.C.shape[0]

        if self.A.shape != (self.n_states, self.n_states):
            raise DimensionMismatchError(f"A must be square, got {self.A.shape}")

        if self.B.shape != (self.n_states, self.n_inputs):
            raise DimensionMismatchError(
                f"B must match states/inputs, got {self.B.shape} (expected {self.n_states}x{self.n_inputs})"
            )

        if self.C.shape != (self.n_outputs, self.n_states):
            raise DimensionMismatchError(
                f"C must match outputs/states, got {self.C.shape} (expected {self.n_outputs}x{self.n_states})"
            )

        if self.D.shape != (self.n_outputs, self.n_inputs):
            raise DimensionMismatchError(
                f"D must match outputs/inputs, got {self.D.shape} (expected {self.n_outputs}x{self.n_inputs})"
            )

    def get_frequency_response(self, omega_range, input_idx=0, output_idx=0):
        """
        Computes the frequency response H(jw) directly from state-space matrices.
        Formula: H(s) = C * (sI - A)^-1 * B + D
        """
        if not (0 <= input_idx < self.n_inputs):
            raise ValueError(f"Invalid input_idx {input_idx}")
        if not (0 <= output_idx < self.n_outputs):
            raise ValueError(f"Invalid output_idx {output_idx}")

        mags = []
        phases = []

        I = np.eye(self.n_states)

        B_j = self.B[:, input_idx : input_idx + 1]
        C_i = self.C[output_idx : output_idx + 1, :]
        D_ij = self.D[output_idx, input_idx]

        for w in omega_range:
            s = 1j * w
            try:
                # Solve (sI - A)x = B for x, which is faster/safer than inv(sI - A)
                term = np.linalg.solve(s * I - self.A, B_j)
                resp_complex = (C_i @ term)[0, 0] + D_ij

                mags.append(20 * np.log10(np.abs(resp_complex)))
                phases.append(np.degrees(np.angle(resp_complex)))
            except np.linalg.LinAlgError:
                # Handle singularities (poles on the jw axis)
                mags.append(np.inf)
                phases.append(0.0)

        return np.array(mags), np.array(phases)
