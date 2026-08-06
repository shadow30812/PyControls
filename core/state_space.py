import logging
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from core.exceptions import DimensionMismatchError

logger = logging.getLogger(__name__)


class StateSpace:
    """
    A container for MIMO Linear Time-Invariant (LTI) Systems.

    This class encapsulates the state-space representation of a system defined by:
        dx/dt = Ax + Bu  (State Equation)
        y     = Cx + Du  (Output Equation)

    It validates the dimensions of the provided matrices to ensure they form a
    consistent system.

    Attributes:
        A (np.ndarray): State transition matrix (n_states x n_states).
        B (np.ndarray): Input matrix (n_states x n_inputs).
        C (np.ndarray): Output matrix (n_outputs x n_states).
        D (np.ndarray): Feedthrough matrix (n_outputs x n_inputs).
        n_states (int): Number of state variables.
        n_inputs (int): Number of control inputs.
        n_outputs (int): Number of measurement outputs.
    """

    def __init__(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        D: ArrayLike,
    ) -> None:
        """
        Initializes the StateSpace system and validates matrix dimensions.

        Args:
            A (array-like): The state matrix.
            B (array-like): The input matrix.
            C (array-like): The output matrix.
            D (array-like): The direct transmission (feedthrough) matrix.

        Raises:
            DimensionMismatchError: If the matrix dimensions are inconsistent with
            each other (e.g., if A is not square, or B/C/D do not match the state dimension).
        """
        self.A: NDArray[np.float64] = np.array(A, dtype=float)
        self.B: NDArray[np.float64] = np.array(B, dtype=float)
        self.C: NDArray[np.float64] = np.array(C, dtype=float)
        self.D: NDArray[np.float64] = np.array(D, dtype=float)

        self.n_states: int = self.A.shape[0]
        self.n_inputs: int = self.B.shape[1]
        self.n_outputs: int = self.C.shape[0]

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

        self._I: NDArray[np.float64] = np.eye(self.n_states)

    def get_frequency_response(
        self, omega_range: ArrayLike, input_idx: int = 0, output_idx: int = 0
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Computes the frequency response H(jw) directly from state-space matrices
        for a specific input-output pair over a range of frequencies.

        The transfer function is calculated as:
            H(s) = C * (sI - A)^-1 * B + D
        where s = jw.

        This method solves the linear system (sI - A)x = B directly rather than
        explicitly inverting the matrix, which is numerically more stable.

        Args:
            omega_range (array-like): A list or array of frequency points (rad/s).
            input_idx (int, optional): Index of the input channel. Defaults to 0.
            output_idx (int, optional): Index of the output channel. Defaults to 0.

        Returns:
            tuple: A tuple (magnitudes, phases) where:
                - magnitudes is a numpy array of magnitude values in dB.
                - phases is a numpy array of phase values in degrees.

        Raises:
            ValueError: If input_idx or output_idx are out of bounds.
        """
        if not (0 <= input_idx < self.n_inputs):
            raise ValueError(f"Invalid input_idx {input_idx}")
        if not (0 <= output_idx < self.n_outputs):
            raise ValueError(f"Invalid output_idx {output_idx}")

        omega: NDArray[np.float64] = np.asarray(omega_range)
        mags: NDArray[np.float64] = np.empty_like(omega, dtype=float)
        phases: NDArray[np.float64] = np.empty_like(omega, dtype=float)

        B_j: NDArray[np.float64] = self.B[:, input_idx : input_idx + 1]
        C_i: NDArray[np.float64] = self.C[output_idx : output_idx + 1, :]
        D_ij: float = self.D[output_idx, input_idx]

        for k, w in enumerate(omega):
            s: complex = 1j * w
            try:
                term: NDArray[np.complexfloating] = np.linalg.solve(
                    s * self._I - self.A, B_j
                )
                resp: complex = (C_i @ term)[0, 0] + D_ij
                mags[k] = 20.0 * np.log10(np.abs(resp))
                phases[k] = np.degrees(np.angle(resp))
            except np.linalg.LinAlgError:
                logger.warning("LinAlg error in frequency response at ω=%s", w)
                mags[k] = np.inf
                phases[k] = 0.0

        return mags, phases
