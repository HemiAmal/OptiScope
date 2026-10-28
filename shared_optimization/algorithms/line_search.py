"""
Line search algorithms: analytical exact line search for quadratic forms,
Armijo sufficient decrease condition, and geometric backtracking.
"""

from typing import Callable, List, Optional, Sequence, Tuple
from shared_optimization.core.vectors import validate_vector, dot_product, vec_add, scalar_mul, Number
from shared_optimization.core.matrices import mat_vec_mul, Matrix

ScalarFunction = Callable[[Sequence[Number]], float]
AcceptanceRule = Callable[[float, List[float], float], bool]


def exact_line_search_quadratic(
    x: Sequence[Number],
    p: Sequence[Number],
    Q: Matrix,
    grad: Optional[Sequence[Number]] = None,
) -> float:
    """
    Computes the exact analytical optimal step length alpha* for a quadratic form.

    Formula:
        alpha* = - <grad f(x), p> / <p, Q * p>

    Parameters:
        x: Current iterate in R^n.
        p: Search direction in R^n.
        Q: Symmetric positive-definite matrix in R^{n x n}.
        grad: Optional precomputed gradient grad f(x) = Q * x.

    Returns:
        float: Exact optimal step size alpha*.
    """
    x_val = validate_vector(x, "x")
    p_val = validate_vector(p, "p")

    # If gradient not provided, compute nabla f(x) = Q * x
    if grad is None:
        g = mat_vec_mul(Q, x_val)
    else:
        g = validate_vector(grad, "grad")

    numerator = -dot_product(g, p_val)
    Qp = mat_vec_mul(Q, p_val)
    denominator = dot_product(p_val, Qp)

    if denominator <= 1e-15:
        raise ValueError(
            f"Exact line search requires positive curvature: <p, Q*p> = {denominator:.2e} <= 0."
        )

    return numerator / denominator


def armijo_condition(
    f: ScalarFunction,
    x: Sequence[Number],
    p: Sequence[Number],
    alpha: float,
    f_x: float,
    grad_x: Sequence[Number],
    c1: float = 1e-4,
) -> bool:
    """
    Evaluates the Armijo sufficient decrease condition:
        f(x + alpha * p) <= f(x) + c_1 * alpha * <grad f(x), p>

    Parameters:
        f: Scalar objective function.
        x: Current iterate in R^n.
        p: Search direction in R^n.
        alpha: Candidate step size.
        f_x: Pre-evaluated function value f(x).
        grad_x: Pre-evaluated gradient vector grad f(x).
        c1: Armijo decrease parameter (default: 1e-4).

    Returns:
        bool: True if sufficient decrease condition holds, False otherwise.
    """
    x_val = validate_vector(x, "x")
    p_val = validate_vector(p, "p")
    g_val = validate_vector(grad_x, "grad_x")

    x_trial = vec_add(x_val, scalar_mul(alpha, p_val))
    f_trial = f(x_trial)
    directional_derivative = dot_product(g_val, p_val)
    threshold = f_x + c1 * alpha * directional_derivative
    return f_trial <= threshold


def backtracking_line_search(
    f: ScalarFunction,
    x: Sequence[Number],
    p: Sequence[Number],
    acceptance_fn: Callable[[float], bool],
    alpha_init: float = 1.0,
    rho: float = 0.5,
    min_alpha: float = 1e-16,
    max_iters: int = 50,
) -> Tuple[float, int]:
    """
    Geometric backtracking line search routine.
    Repeatedly contracts alpha <- rho * alpha until acceptance_fn(alpha) is True.
    """
    if alpha_init <= 0:
        raise ValueError(f"Initial step size alpha_init must be positive, got {alpha_init}.")
    if not (0.0 < rho < 1.0):
        raise ValueError(f"Contraction factor rho must be in (0, 1), got {rho}.")

    alpha = alpha_init
    evals = 0

    while evals < max_iters and alpha >= min_alpha:
        evals += 1
        if acceptance_fn(alpha):
            return alpha, evals
        alpha *= rho

    # Return smallest explored alpha if acceptance condition failed
    return alpha, evals
