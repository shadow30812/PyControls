import cmath
import math
import re
from typing import Callable

import numpy as np

# Constants
TOL = 1e-15
h = 1e-12
iter_max = 1000


# --- 1. The Parser ---
def implicit_mul(expr: str) -> str:
    """Convert '3x' to '3*x'."""
    expr = re.sub(r"(?<=[0-9\)])(?=[A-Za-z\(])", "*", expr)
    expr = re.sub(r"(?<=[A-Za-z\)])(?=[0-9\(])", "*", expr)
    return expr


def preprocess_power(expr: str) -> str:
    """Convert '^' to '**'."""
    return re.sub(r"(?<=\w)\^(?=\w|\()", "**", expr)


def make_func(
    expr_string: str, var_name: str = "t"
) -> Callable[[float | complex], float | complex]:
    """
    Creates a function from a string (Scalar Math).
    Supports complex inputs for Complex Step Differentiation.
    """
    expr = preprocess_power(implicit_mul(expr_string))

    # Allow both math (real) and cmath (complex) functions
    safe_locals = {}
    safe_locals.update(
        {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    )
    safe_locals.update(
        {k: getattr(cmath, k) for k in dir(cmath) if not k.startswith("_")}
    )

    def f(value):
        safe_locals[var_name] = value
        try:
            return eval(expr, {"__builtins__": {}}, safe_locals)
        except Exception:
            return 0.0

    return f


def make_system_func(expr_string: str) -> Callable[[np.ndarray, float], np.ndarray]:
    """
    Parses a math string into a function f(x, u) (Vector/Numpy Math).
    Example: "-0.5*x + sin(u)"
    """
    expr = preprocess_power(implicit_mul(expr_string))

    # Inject all numpy functions so user doesn't need 'np.' prefix
    safe_locals = {"pi": np.pi, "e": np.e}
    for name in dir(np):
        if not name.startswith("_"):
            safe_locals[name] = getattr(np, name)

    def f(x, u):
        safe_locals["x"] = x
        safe_locals["u"] = u
        try:
            res = eval(expr, {"__builtins__": {}}, safe_locals)
            if np.isscalar(res):
                return np.full_like(x, res)
            return np.array(res)
        except Exception:
            return np.zeros_like(x)

    return f


# --- 2. The Differentiation Engine ---
class Differentiation:
    """
    Handler of real and complex numerical differentiation.
    Uses Complex Step Differentiation when possible for maximum accuracy.
    """

    def real_diff(self, func: Callable[..., float], point: float) -> float:
        try:
            # 1. Try Complex Step Differentiation (High Accuracy)
            arg = complex(point, h)
            func_result = func(arg)

            # Check if imaginary part propagated
            imag_part = complex(func_result).imag

            if abs(imag_part) > 0.0:
                return imag_part / h
            else:
                # Fallback if function was non-analytic (e.g. abs(), angle())
                raise ValueError("Complex step did not propagate")

        except Exception:
            # 2. Fallback to Finite Difference (Robustness)
            try:
                return (func(point + h) - func(point - h)) / (2 * h)
            except Exception:
                return 0.0


# --- 3. The Root Finder ---
def find_root(func: Callable, guess0: float) -> float:
    """Newton-Raphson solver."""
    guess = guess0
    diff_tool = Differentiation()

    for _ in range(iter_max):
        f_val = func(guess)
        if isinstance(f_val, complex):
            f_val = f_val.real

        f_prime = diff_tool.real_diff(func, guess)

        if abs(f_prime) < TOL:
            break

        new_guess = guess - f_val / f_prime
        if abs(new_guess - guess) < TOL:
            return new_guess
        guess = new_guess

    return guess
