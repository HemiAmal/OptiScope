"""
Numerical derivatives and finite-difference verification utilities.
"""

from typing import Callable, Dict, List, Optional, Sequence, Union
from shared_optimization.core.vectors import validate_vector, Number, Vector
from shared_optimization.core.matrices import Matrix, validate_matrix

ScalarFunction = Callable[[Sequence[Number]], float]
GradientFunction = Callable[[Sequence[Number]], List[float]]
HessianFunction = Callable[[Sequence[Number]], Matrix]


def finite_difference_gradient(
    f: ScalarFunction, x: Sequence[Number], h: float = 1e-6
) -> List[float]:
    """
    Compute gradient of scalar function f at point x using central finite differences.
    """
    if h <= 0:
        raise ValueError(f"Perturbation step size h must be positive, got {h}.")

    x_val = validate_vector(x, "x")
    n = len(x_val)
    grad = [0.0] * n

    for i in range(n):
        x_plus = list(x_val)
        x_minus = list(x_val)
        x_plus[i] += h
        x_minus[i] -= h

        f_plus = f(x_plus)
        f_minus = f(x_minus)

        grad[i] = (f_plus - f_minus) / (2.0 * h)

    return grad


def finite_difference_hessian(
    f: ScalarFunction, x: Sequence[Number], h: float = 1e-4
) -> Matrix:
    """
    Compute Hessian matrix of f at point x using central differences of function values.
    """
    if h <= 0:
        raise ValueError(f"Perturbation step size h must be positive, got {h}.")

    x_val = validate_vector(x, "x")
    n = len(x_val)
    f_center = f(x_val)

    H: Matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            if i == j:
                x_plus = list(x_val)
                x_minus = list(x_val)
                x_plus[i] += h
                x_minus[i] -= h
                val = (f(x_plus) - 2.0 * f_center + f(x_minus)) / (h * h)
                H[i][i] = val
            else:
                p_pp = list(x_val); p_pp[i] += h; p_pp[j] += h
                p_pm = list(x_val); p_pm[i] += h; p_pm[j] -= h
                p_mp = list(x_val); p_mp[i] -= h; p_mp[j] += h
                p_mm = list(x_val); p_mm[i] -= h; p_mm[j] -= h

                val = (f(p_pp) - f(p_pm) - f(p_mp) + f(p_mm)) / (4.0 * h * h)
                H[i][j] = val
                H[j][i] = val

    return H


def finite_difference_hessian_from_grad(
    grad_f: GradientFunction, x: Sequence[Number], h: float = 1e-5
) -> Matrix:
    """
    Compute Hessian matrix by taking central finite differences of the gradient function.
    """
    if h <= 0:
        raise ValueError(f"Perturbation step size h must be positive, got {h}.")

    x_val = validate_vector(x, "x")
    n = len(x_val)
    H: Matrix = [[0.0] * n for _ in range(n)]

    for j in range(n):
        x_plus = list(x_val)
        x_minus = list(x_val)
        x_plus[j] += h
        x_minus[j] -= h

        g_plus = grad_f(x_plus)
        g_minus = grad_f(x_minus)

        if len(g_plus) != n or len(g_minus) != n:
            raise ValueError(f"Gradient output dimension must match x dimension {n}.")

        for i in range(n):
            H[i][j] = (g_plus[i] - g_minus[i]) / (2.0 * h)

    # Symmetrize
    for i in range(n):
        for j in range(i + 1, n):
            avg = 0.5 * (H[i][j] + H[j][i])
            H[i][j] = avg
            H[j][i] = avg

    return H


def verify_gradient(
    f: ScalarFunction,
    analytical_grad_fn: GradientFunction,
    x: Sequence[Number],
    h: float = 1e-6,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Dict[str, Union[bool, float, List[float]]]:
    """
    Verify analytical gradient against central finite differences.
    Tolerance check: |approx_i - exact_i| <= atol + rtol * |exact_i|
    """
    x_val = validate_vector(x, "x")
    analytical = analytical_grad_fn(x_val)
    numerical = finite_difference_gradient(f, x_val, h=h)

    n = len(x_val)
    if len(analytical) != n:
        raise ValueError(f"Analytical gradient length {len(analytical)} != x length {n}.")

    max_abs_err = 0.0
    max_rel_err = 0.0
    all_passed = True

    for i in range(n):
        exact = analytical[i]
        approx = numerical[i]
        abs_err = abs(approx - exact)
        rel_err = abs_err / (abs(exact) + 1e-15)

        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err

        if abs_err > (atol + rtol * abs(exact)):
            all_passed = False

    return {
        "passed": all_passed,
        "analytical": analytical,
        "numerical": numerical,
        "max_abs_error": max_abs_err,
        "max_rel_error": max_rel_err,
    }


def verify_hessian(
    f: ScalarFunction,
    analytical_hess_fn: HessianFunction,
    x: Sequence[Number],
    grad_f: Optional[GradientFunction] = None,
    h: float = 1e-4,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Dict[str, Union[bool, float, Matrix]]:
    """
    Verify analytical Hessian against finite-difference Hessian.
    If grad_f is provided, also checks gradient finite differences.
    """
    x_val = validate_vector(x, "x")
    analytical = validate_matrix(analytical_hess_fn(x_val), "analytical Hessian")
    numerical = finite_difference_hessian(f, x_val, h=h)

    n = len(x_val)
    if len(analytical) != n or len(analytical[0]) != n:
        raise ValueError(f"Analytical Hessian shape must be {n}x{n}.")

    max_abs_err = 0.0
    max_rel_err = 0.0
    all_passed = True

    for i in range(n):
        for j in range(n):
            exact = analytical[i][j]
            approx = numerical[i][j]
            abs_err = abs(approx - exact)
            rel_err = abs_err / (abs(exact) + 1e-15)

            if abs_err > max_abs_err:
                max_abs_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err

            if abs_err > (atol + rtol * abs(exact)):
                all_passed = False

    # Also test gradient finite difference if grad_f provided
    grad_passed = True
    if grad_f is not None:
        H_grad = finite_difference_hessian_from_grad(grad_f, x_val, h=h)
        for i in range(n):
            for j in range(n):
                exact = analytical[i][j]
                approx = H_grad[i][j]
                if abs(approx - exact) > (atol + rtol * abs(exact)):
                    grad_passed = False

    return {
        "passed": all_passed and grad_passed,
        "analytical": analytical,
        "numerical": numerical,
        "max_abs_error": max_abs_err,
        "max_rel_error": max_rel_err,
    }
