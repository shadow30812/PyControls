class PyControlsError(Exception):
    """Base class for all exceptions in PyControls."""

    pass


class DimensionMismatchError(ValueError, PyControlsError):
    """
    Raised when matrix or vector dimensions are inconsistent with the
    mathematical operation (e.g., A matrix is not square, B doesn't match A).
    Inherits from ValueError for backward compatibility.
    """

    pass


class SingularMatrixError(ArithmeticError, PyControlsError):
    """
    Raised when a matrix is singular and cannot be inverted or solved.
    """

    pass


class ConvergenceError(RuntimeError, PyControlsError):
    """
    Raised when an iterative numerical method (Newton, Brent, RK45)
    fails to converge within the maximum number of iterations or tolerances.
    """

    pass


class UnstableSystemError(PyControlsError):
    """
    Raised when an operation requires a stable system but an unstable one was provided.
    """

    pass
