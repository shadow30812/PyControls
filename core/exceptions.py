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


class InvalidParameterError(ValueError, PyControlsError):
    """
    Raised when a physical parameter is invalid (e.g., negative mass, zero inertia).
    """

    pass


class SolverError(RuntimeError, PyControlsError):
    """
    Raised when a numerical solver encounters a critical failure (e.g., step size underflow).
    """

    pass


class ControllerConfigError(ValueError, PyControlsError):
    """
    Raised when a controller configuration is invalid (e.g., missing gains, invalid limits).
    """

    pass
