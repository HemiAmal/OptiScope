"""
Gradient descent algorithms including fixed step size, exact quadratic line search,
and Armijo backtracking line search.
"""

import time
from typing import Callable, List, Optional, Sequence
from shared_optimization.core.vectors import validate_vector, vec_add, scalar_mul, norm_l2, dot_product, Number
from shared_optimization.core.matrices import mat_vec_mul, Matrix
from shared_optimization.core.convergence import (
    OptimizationResult,
    IterationRecord,
    ConvergenceCriterion,
    check_convergence,
    check_finite,
)
from shared_optimization.algorithms.line_search import (
    exact_line_search_quadratic,
    backtracking_line_search,
    armijo_condition,
)

ScalarFunction = Callable[[Sequence[Number]], float]
GradientFunction = Callable[[Sequence[Number]], List[float]]


def gradient_descent_fixed(
    f: ScalarFunction,
    grad_f: GradientFunction,
    x0: Sequence[Number],
    learning_rate: float = 0.01,
    criterion: Optional[ConvergenceCriterion] = None,
) -> OptimizationResult:
    """
    Minimizes a scalar function using gradient descent with fixed step size.

    Update Rule:
        x_{k+1} = x_k - alpha * grad f(x_k)

    Parameters:
        f: Objective function.
        grad_f: Gradient function.
        x0: Initial parameter vector.
        learning_rate: Step size alpha > 0.
        criterion: Convergence tolerances and iteration cap.

    Returns:
        OptimizationResult: Convergence metrics and trajectory history.
    """
    start_time = time.time()
    crit = criterion if criterion is not None else ConvergenceCriterion()
    x = validate_vector(x0, "x0")
    check_finite(x, "x0")

    history: List[IterationRecord] = []
    func_evals = 0
    grad_evals = 0

    f_val = f(x)
    func_evals += 1
    g = grad_f(x)
    grad_evals += 1
    g_norm = norm_l2(g)

    history.append(IterationRecord(iteration=0, x=list(x), fun=f_val, grad_norm=g_norm, step_size=None))

    converged = False
    status_msg = "Maximum iterations reached"

    for k in range(1, crit.max_iters + 1):
        # Check convergence on current gradient
        conv = check_convergence(grad_norm=g_norm, criterion=crit)
        if conv is not None:
            converged = True
            status_msg = conv
            break

        # Steepest descent step: x_{k+1} = x_k - alpha * g
        step = scalar_mul(-learning_rate, g)
        x_next = vec_add(x, step)
        check_finite(x_next, f"iterate_{k}")

        f_next = f(x_next)
        func_evals += 1
        g_next = grad_f(x_next)
        grad_evals += 1
        g_next_norm = norm_l2(g_next)
        step_norm = norm_l2(step)
        delta_f = f_next - f_val

        history.append(
            IterationRecord(
                iteration=k,
                x=list(x_next),
                fun=f_next,
                grad_norm=g_next_norm,
                step_size=learning_rate,
            )
        )

        # Check step-based convergence
        conv_step = check_convergence(
            grad_norm=g_next_norm,
            step_norm=step_norm,
            delta_f=delta_f,
            f_current=f_next,
            criterion=crit,
        )
        if conv_step is not None:
            converged = True
            status_msg = conv_step
            x, f_val, g_norm = x_next, f_next, g_next_norm
            break

        x, f_val, g, g_norm = x_next, f_next, g_next, g_next_norm

    exec_time = time.time() - start_time
    return OptimizationResult(
        x=x,
        fun=f_val,
        grad_norm=g_norm,
        iterations=len(history) - 1,
        func_evals=func_evals,
        grad_evals=grad_evals,
        hess_evals=0,
        converged=converged,
        status_message=status_msg,
        history=history,
        execution_time=exec_time,
    )


def gradient_descent_exact_quadratic(
    x0: Sequence[Number],
    Q: Matrix,
    criterion: Optional[ConvergenceCriterion] = None,
) -> OptimizationResult:
    """
    Gradient descent with analytical exact line search for quadratic objectives.

    Minimizes f(x) = 0.5 * x^T * Q * x by computing the exact analytical optimal step size:
        alpha_k = <g_k, g_k> / <g_k, Q * g_k>
        x_{k+1} = x_k - alpha_k * g_k
    """
    start_time = time.time()
    crit = criterion if criterion is not None else ConvergenceCriterion()
    x = validate_vector(x0, "x0")

    def f_quad(v):
        Qv = mat_vec_mul(Q, v)
        return 0.5 * dot_product(v, Qv)

    def grad_quad(v):
        return mat_vec_mul(Q, v)

    history: List[IterationRecord] = []
    func_evals = 0
    grad_evals = 0

    f_val = f_quad(x)
    func_evals += 1
    g = grad_quad(x)
    grad_evals += 1
    g_norm = norm_l2(g)

    history.append(IterationRecord(iteration=0, x=list(x), fun=f_val, grad_norm=g_norm, step_size=None))

    converged = False
    status_msg = "Maximum iterations reached"

    for k in range(1, crit.max_iters + 1):
        if g_norm < crit.grad_tol:
            converged = True
            status_msg = f"Converged: gradient norm {g_norm:.2e} < {crit.grad_tol:.2e}"
            break

        # Search direction p = -g
        p = scalar_mul(-1.0, g)
        alpha = exact_line_search_quadratic(x, p, Q, grad=g)

        step = scalar_mul(alpha, p)
        x_next = vec_add(x, step)
        f_next = f_quad(x_next)
        func_evals += 1
        g_next = grad_quad(x_next)
        grad_evals += 1
        g_next_norm = norm_l2(g_next)
        step_norm = norm_l2(step)
        delta_f = f_next - f_val

        history.append(
            IterationRecord(
                iteration=k,
                x=list(x_next),
                fun=f_next,
                grad_norm=g_next_norm,
                step_size=alpha,
            )
        )

        conv = check_convergence(
            grad_norm=g_next_norm,
            step_norm=step_norm,
            delta_f=delta_f,
            f_current=f_next,
            criterion=crit,
        )
        if conv is not None:
            converged = True
            status_msg = conv
            x, f_val, g_norm = x_next, f_next, g_next_norm
            break

        x, f_val, g, g_norm = x_next, f_next, g_next, g_next_norm

    exec_time = time.time() - start_time
    return OptimizationResult(
        x=x,
        fun=f_val,
        grad_norm=g_norm,
        iterations=len(history) - 1,
        func_evals=func_evals,
        grad_evals=grad_evals,
        hess_evals=0,
        converged=converged,
        status_message=status_msg,
        history=history,
        execution_time=exec_time,
    )


def gradient_descent_armijo(
    f: ScalarFunction,
    grad_f: GradientFunction,
    x0: Sequence[Number],
    alpha_init: float = 1.0,
    rho: float = 0.5,
    c1: float = 1e-4,
    criterion: Optional[ConvergenceCriterion] = None,
) -> OptimizationResult:
    """
    Gradient descent with Armijo backtracking line search.

    Ensures sufficient decrease at each step via:
        f(x_k + alpha * p_k) <= f(x_k) + c1 * alpha * <grad f(x_k), p_k>
    """
    start_time = time.time()
    crit = criterion if criterion is not None else ConvergenceCriterion()
    x = validate_vector(x0, "x0")
    check_finite(x, "x0")

    history: List[IterationRecord] = []
    func_evals = 0
    grad_evals = 0

    f_val = f(x)
    func_evals += 1
    g = grad_f(x)
    grad_evals += 1
    g_norm = norm_l2(g)

    history.append(IterationRecord(iteration=0, x=list(x), fun=f_val, grad_norm=g_norm, step_size=None))

    converged = False
    status_msg = "Maximum iterations reached"

    for k in range(1, crit.max_iters + 1):
        if g_norm < crit.grad_tol:
            converged = True
            status_msg = f"Converged: gradient norm {g_norm:.2e} < {crit.grad_tol:.2e}"
            break

        # Search direction p = -g (steepest descent)
        p = scalar_mul(-1.0, g)

        # Backtracking with Armijo acceptance test
        def armijo_check(trial_alpha: float) -> bool:
            return armijo_condition(
                f=f,
                x=x,
                p=p,
                alpha=trial_alpha,
                f_x=f_val,
                grad_x=g,
                c1=c1,
            )

        alpha, bt_evals = backtracking_line_search(
            f=f,
            x=x,
            p=p,
            acceptance_fn=armijo_check,
            alpha_init=alpha_init,
            rho=rho,
        )
        func_evals += bt_evals

        # Apply update x_{k+1} = x_k + alpha * p
        step = scalar_mul(alpha, p)
        x_next = vec_add(x, step)
        check_finite(x_next, f"iterate_{k}")

        f_next = f(x_next)
        func_evals += 1
        g_next = grad_f(x_next)
        grad_evals += 1
        g_next_norm = norm_l2(g_next)
        step_norm = norm_l2(step)
        delta_f = f_next - f_val

        history.append(
            IterationRecord(
                iteration=k,
                x=list(x_next),
                fun=f_next,
                grad_norm=g_next_norm,
                step_size=alpha,
            )
        )

        conv = check_convergence(
            grad_norm=g_next_norm,
            step_norm=step_norm,
            delta_f=delta_f,
            f_current=f_next,
            criterion=crit,
        )
        if conv is not None:
            converged = True
            status_msg = conv
            x, f_val, g_norm = x_next, f_next, g_next_norm
            break

        x, f_val, g, g_norm = x_next, f_next, g_next, g_next_norm

    exec_time = time.time() - start_time
    return OptimizationResult(
        x=x,
        fun=f_val,
        grad_norm=g_norm,
        iterations=len(history) - 1,
        func_evals=func_evals,
        grad_evals=grad_evals,
        hess_evals=0,
        converged=converged,
        status_message=status_msg,
        history=history,
        execution_time=exec_time,
    )
