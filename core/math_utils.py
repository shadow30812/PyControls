import cmath
import math
import re
from typing import Callable, Optional

import numpy as np

TOL = 1e-12
hc = 1e-12
hf = 1e-6
ITER_MAX = 100


def implicit_mul(expr: str) -> str:
    """
    Inserts explicit multiplication signs for implicit multiplication.
    Example: '3x' -> '3*x', 'sin(x)2' -> 'sin(x)*2'.
    """
    expr = re.sub(r"(?<=[0-9\)])\s*(?=[A-Za-z\(])", "*", expr)
    expr = re.sub(r"(?<=[A-Za-z\)])\s*(?=[0-9])", "*", expr)
    return expr


def preprocess_power(expr: str) -> str:
    """Converts caret power syntax (x^2) to Python syntax (x**2)."""
    return re.sub(r"(?<=\w)\^(?=\w|\()", "**", expr)


def make_func(
    expr_string: str, var_name: str = "t"
) -> Callable[[float | complex], float | complex]:
    """
    Compiles a string expression into a callable Python function.
    The resulting function supports both float and complex arguments,
    enabling Complex Step Differentiation.

    Args:
        expr_string: Mathematical expression (e.g., "sin(t) + t^2").
        var_name: The independent variable name.
    """
    expr = preprocess_power(implicit_mul(expr_string))
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
    Compiles a string expression into a state-space function f(t, x, u).
    Supports NumPy vector operations.

    Args:
        expr_string: Expression returning the derivative vector (e.g. "[x[1], -x[0]]").
    """
    expr = preprocess_power(implicit_mul(expr_string))
    safe_locals = {"pi": np.pi, "e": np.e}
    for name in dir(np):
        if not name.startswith("_"):
            safe_locals[name] = getattr(np, name)

    def f(t, x, u=0.0):
        safe_locals["t"] = t
        safe_locals["x"] = x
        safe_locals["u"] = u
        try:
            res = eval(expr, {"__builtins__": {}}, safe_locals)
            if np.isscalar(res):
                return np.full_like(x, res)
            return np.array(res)
        except Exception as e:
            print(f"DEBUG: Eq Eval Error: {e} | Parsed Expr: {expr}")
            return np.zeros_like(x)

    return f


class Differentiation:
    """
    Helper class for computing derivatives numerically.
    Prioritizes Complex Step Differentiation for high accuracy,
    falling back to Finite Difference if the function doesn't support complex types.
    """

    def real_diff(self, func: Callable[..., float], point: float) -> float:
        try:
            arg = complex(point, hc)
            func_result = func(arg)
            imag_part = complex(func_result).imag
            if abs(imag_part) > 0.0:
                return imag_part / hc
            else:
                raise ValueError("Complex step did not propagate")
        except Exception:
            try:
                return (func(point + hf) - func(point - hf)) / (2 * hf)
            except Exception:
                return 0.0


class Root:
    """
    Collection of robust root-finding algorithms.
    """

    def brent_root(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-12,
        f_tol: float = 1e-12,
        maxiter: int = 100,
    ) -> float:
        """
        Finds a root of f(x) = 0 in the interval [a, b] using Brent's Method.
        Combines Bisection, Secant, and Inverse Quadratic Interpolation.
        Requires f(a) and f(b) to have opposite signs.
        """
        fa = f(a)
        fb = f(b)

        if math.isnan(fa) or math.isnan(fb):
            raise ValueError("Function returned NaN at initial endpoints.")

        if fa * fb > 0:
            raise ValueError(f"Root is not bracketed: f({a})={fa}, f({b})={fb}")

        if abs(fa) <= f_tol:
            return a
        if abs(fb) <= f_tol:
            return b

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        c = a
        fc = fa
        d = a
        mflag = True
        iter_count = 0

        while iter_count < maxiter:
            iter_count += 1

            if abs(b - a) <= tol or abs(fb) <= f_tol:
                return b

            s = None
            try:
                if fa != fc and fb != fc:
                    s = (a * fb * fc) / ((fa - fb) * (fa - fc))
                    s += (b * fa * fc) / ((fb - fa) * (fb - fc))
                    s += (c * fa * fb) / ((fc - fa) * (fc - fb))
                else:
                    denom = fb - fa
                    if denom == 0:
                        s = (a + b) / 2.0
                    else:
                        s = b - fb * (b - a) / denom
            except ZeroDivisionError:
                s = (a + b) / 2.0

            if a > b:
                a, b = b, a
                fa, fb = fb, fa

            cond1 = not ((3 * a + b) / 4 < s < b)
            cond2 = mflag and (abs(s - b) >= abs(b - c) / 2)
            cond3 = (not mflag) and (abs(s - b) >= abs(c - d) / 2)
            cond4 = mflag and (abs(b - c) < tol)
            cond5 = (not mflag) and (abs(c - d) < tol)

            if cond1 or cond2 or cond3 or cond4 or cond5 or s is None or math.isnan(s):
                s = (a + b) / 2.0
                mflag = True
            else:
                mflag = False

            fs = f(s)
            if math.isnan(fs):
                raise ValueError(f"Function returned NaN at s={s}")

            d = c
            c = b
            fc = fb

            if fa * fs < 0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs

            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa

        return b

    def newton_root(
        self,
        func: Callable[[float], float],
        guess: float,
        tol: float = 1e-12,
        maxiter: int = 100,
    ) -> float:
        """
        Finds a root using the Newton-Raphson method.
        Calculates derivatives automatically using the Differentiation helper.
        Useful when the root is not bracketed (e.g. roots of parabolas touching zero).
        """
        x = guess
        diff_tool = Differentiation()

        for _ in range(maxiter):
            f_val = func(x)
            if isinstance(f_val, complex):
                f_val = f_val.real

            if abs(f_val) < tol:
                return x

            f_prime = diff_tool.real_diff(func, x)

            if abs(f_prime) < 1e-15:
                break

            x_new = x - f_val / f_prime
            if abs(x_new - x) < tol:
                return x_new
            x = x_new

        return x

    def find_root(
        self,
        func: Callable[[float], float],
        x0: float,
        x1: Optional[float] = None,
        tol: float = 1e-12,
    ) -> float:
        """
        Hybrid Root Finder.

        Strategy:
        1. If two points [x0, x1] are provided, try Brent's method (bracketed).
        2. If Brent's fails (bad bracket) or only one point provided, fall back to Newton-Raphson.
        """
        if x1 is not None:
            try:
                return self.brent_root(func, x0, x1, tol=tol, f_tol=tol)
            except ValueError:
                pass

        return self.newton_root(func, x0, tol=tol)
