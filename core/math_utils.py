import cmath
import math
import re
from typing import Callable, Optional

import numpy as np

from core.exceptions import ConvergenceError

TOL = 1e-12
hc = 1e-12
hf = 1e-6
ITER_MAX = 100

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


def implicit_mul(expr: str) -> str:
    """Inserts explicit multiplication signs for implicit multiplication."""
    _IMPLICIT_1 = re.compile(r"(?<=[0-9\)])\s*(?=[A-Za-z\(])")
    _IMPLICIT_2 = re.compile(r"(?<=[A-Za-z\)])\s*(?=[0-9])")
    expr = re.sub(_IMPLICIT_1, "*", expr)
    expr = re.sub(_IMPLICIT_2, "*", expr)
    return expr


def preprocess_power(expr: str) -> str:
    """Converts caret power syntax (x^2) to Python syntax (x**2)."""
    _PRE = re.compile(r"(?<=\w)\^(?=\w|\()")
    return re.sub(_PRE, "**", expr)


def make_func(
    expr_string: str, var_name: str = "t"
) -> Callable[[float | complex], float | complex]:
    """Compiles a string expression into a callable Python function."""
    expr = preprocess_power(implicit_mul(expr_string))
    safe_locals = {}
    safe_locals.update(
        {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    )
    safe_locals.update(
        {k: getattr(cmath, k) for k in dir(cmath) if not k.startswith("_")}
    )

    code = compile(expr, "<expr>", "eval")

    def f(value):
        safe_locals[var_name] = value
        try:
            return eval(code, {"__builtins__": {}}, safe_locals)
        except Exception:
            return 0.0

    return f


def make_system_func(expr_string: str):
    """Compiles a string expression into a state-space function f(t, x, u)."""

    expr = preprocess_power(implicit_mul(expr_string))
    safe_locals = {
        name: getattr(np, name) for name in dir(np) if not name.startswith("_")
    }

    safe_locals.update({"pi": np.pi, "e": np.e})
    code = compile(expr, "<system_func>", "eval")

    def f(t, x, u=0.0):
        loc = safe_locals
        loc["t"] = t
        loc["x"] = x
        loc["u"] = u
        try:
            res = eval(code, {"__builtins__": {}}, loc)
            return np.asarray(res, dtype=float)
        except Exception:
            return np.zeros_like(x)

    return f


class Differentiation:
    def real_diff(self, func: Callable[..., float], point: float) -> float:
        try:
            arg = complex(point, hc)
            func_result = func(arg)
            imag_part = complex(func_result).imag
            if imag_part != 0.0:
                return imag_part / hc
            else:
                raise ValueError("Complex step did not propagate")
        except Exception:
            try:
                return (func(point + hf) - func(point - hf)) / (2 * hf)
            except Exception:
                return 0.0


def jacobian(func, x, *args):
    """
    Computes the Jacobian of a vector-valued function using complex-step differentiation.

    Args:
        func: Callable f(x, *args) -> array_like
        x: 1D numpy array
        *args: Additional arguments passed to func

    Returns:
        J: Jacobian matrix (m x n)
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    eps = hc

    y0 = np.asarray(func(x, *args), dtype=float)
    m = y0.size

    J = np.zeros((m, n), dtype=float)

    try:
        x_c = x.astype(complex)
        for i in range(n):
            x_pert = x_c.copy()
            x_pert[i] += 1j * eps
            y_pert = func(x_pert, *args)
            J[:, i] = np.imag(y_pert) / eps
        return J
    except Exception:
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = hf
            f_plus = func(x + dx, *args)
            f_minus = func(x - dx, *args)
            J[:, i] = (f_plus - f_minus) / (2 * hf)
        return J


class Root:
    def brent_root(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-12,
        f_tol: float = 1e-12,
        maxiter: int = 100,
    ) -> float:
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

        raise ConvergenceError(
            f"Brent's method failed to converge after {maxiter} iterations"
        )

    def newton_root(
        self,
        func: Callable[[float], float],
        guess: float,
        tol: float = 1e-12,
        maxiter: int = 100,
    ) -> float:
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

        raise ConvergenceError(
            f"Newton's method failed to converge after {maxiter} iterations"
        )

    def find_root(
        self,
        func: Callable[[float], float],
        x0: float,
        x1: Optional[float] = None,
        tol: float = 1e-12,
    ) -> float:
        if x1 is not None:
            try:
                return self.brent_root(func, x0, x1, tol=tol, f_tol=tol)
            except (ValueError, ConvergenceError):
                pass

        try:
            return self.newton_root(func, x0, tol=tol)
        except ConvergenceError:
            return x0
