"""
Newton's method optimization algorithms: pure Newton, damped Newton with line search,
and quadratic one-step convergence verification.
"""

import time
from typing import Callable, Dict, List, Optional, Sequence, Union
from shared_optimization.core.vectors import validate_vector, vec_add, scalar_mul, norm_l2, dot_product, Number
from shared_optimization.core.matrices import validate_matrix, mat_vec_mul, Matrix
from shared_optimization.core.linear_systems import solve_linear_system, SingularMatrixError
from shared_optimization.core.convergence import (
    OptimizationResult,
    IterationRecord,
    ConvergenceCriterion,
    check_convergence,
    check_finite,
)
from shared_optimization.algorithms.line_search import backtracking_line_search, armijo_condition

ScalarFunction = Callable[[Sequence[Number]], float]
GradientFunction = Callable[[Sequence[Number]], List[float]]
HessianFunction = Callable[[Sequence[Number]], Matrix]


def newton_method(
    f: ScalarFunction,
    grad_f: GradientFunction,
    hess_f: HessianFunction,
    x0: Sequence[Number],
    criterion: Optional[ConvergenceCriterion] = None,
) -> OptimizationResult:
    """
    Minimizes a function using pure Newton's method:
        H_k * p_k = - grad f(x_k)  (solved via Gaussian elimination with partial pivoting)
        x_{k+1} = x_k + p_k
    """
    start_time = time.time()
    crit = criterion if criterion is not None else ConvergenceCriterion()
    x = validate_vector(x0, "x0")
    check_finite(x, "x0")

    history: List[IterationRecord] = []
    func_evals = 0
    grad_evals = 0
    hess_evals = 0

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

        # Compute analytical Hessian H_k
        H = hess_f(x)
        hess_evals += 1

        # Solve H_k * p_k = -g_k
        neg_g = scalar_mul(-1.0, g)
        try:
            p = solve_linear_system(H, neg_g)
        except SingularMatrixError as e:
            status_msg = f"Terminated: Hessian is singular at iteration {k} ({e})"
            break

        # Full Newton update (step size = 1.0)
        x_next = vec_add(x, p)
        check_finite(x_next, f"iterate_{k}")

        f_next = f(x_next)
        func_evals += 1
        g_next = grad_f(x_next)
        grad_evals += 1
        g_next_norm = norm_l2(g_next)
        step_norm = norm_l2(p)
        delta_f = f_next - f_val

        history.append(
            IterationRecord(
                iteration=k,
                x=list(x_next),
                fun=f_next,
                grad_norm=g_next_norm,
                step_size=1.0,
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
        hess_evals=hess_evals,
        converged=converged,
        status_message=status_msg,
        history=history,
        execution_time=exec_time,
    )


def damped_newton_method(
    f: ScalarFunction,
    grad_f: GradientFunction,
    hess_f: HessianFunction,
    x0: Sequence[Number],
    alpha_init: float = 1.0,
    rho: float = 0.5,
    c1: float = 1e-4,
    criterion: Optional[ConvergenceCriterion] = None,
) -> OptimizationResult:
    """
    Damped Newton method with Armijo backtracking line search along the Newton direction.
    """
    start_time = time.time()
    crit = criterion if criterion is not None else ConvergenceCriterion()
    x = validate_vector(x0, "x0")
    check_finite(x, "x0")

    history: List[IterationRecord] = []
    func_evals = 0
    grad_evals = 0
    hess_evals = 0

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

        H = hess_f(x)
        hess_evals += 1

        neg_g = scalar_mul(-1.0, g)
        try:
            p = solve_linear_system(H, neg_g)
        except SingularMatrixError as e:
            status_msg = f"Terminated: Hessian is singular at iteration {k} ({e})"
            break

        # Check descent direction: <g, p> must be < 0
        direct_deriv = dot_product(g, p)
        if direct_deriv >= 0:
            # If not descent (e.g. non-convex region), revert to steepest descent p = -g
            p = neg_g

        # Backtracking line search with Armijo condition
        def armijo_check(trial_alpha: float) -> bool:
            return armijo_condition(f, x, p, trial_alpha, f_val, g, c1=c1)

        alpha, bt_evals = backtracking_line_search(
            f=f,
            x=x,
            p=p,
            acceptance_fn=armijo_check,
            alpha_init=alpha_init,
            rho=rho,
        )
        func_evals += bt_evals

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
        hess_evals=hess_evals,
        converged=converged,
        status_message=status_msg,
        history=history,
        execution_time=exec_time,
    )


def verify_newton_one_step(
    Q: Matrix,
    c: Sequence[Number],
    x0: Sequence[Number],
    tol: float = 1e-10,
) -> Dict[str, Union[bool, float, List[float], OptimizationResult]]:
    """
    Demonstrates that Newton's method on a strictly convex quadratic
    f(x) = 0.5 * x^T * Q * x + c^T * x converges to the global minimizer
    in exactly one step from any starting point x0.
    """
    x0_val = validate_vector(x0, "x0")
    c_val = validate_vector(c, "c")
    Q_val = validate_matrix(Q, "Q")

    # True theoretical minimizer: solve Q * x* = -c
    neg_c = scalar_mul(-1.0, c_val)
    x_star = solve_linear_system(Q_val, neg_c)

    def f_quad(x):
        Qx = mat_vec_mul(Q_val, x)
        return 0.5 * dot_product(x, Qx) + dot_product(c_val, x)

    def grad_quad(x):
        Qx = mat_vec_mul(Q_val, x)
        return vec_add(Qx, c_val)

    def hess_quad(x):
        return Q_val

    # Run Newton with max_iters = 1
    crit = ConvergenceCriterion(max_iters=1, grad_tol=tol)
    res = newton_method(f_quad, grad_quad, hess_quad, x0_val, criterion=crit)

    # After 1 step, x_1 must equal x_star
    x_1 = res.x
    error = norm_l2(vec_add(x_1, scalar_mul(-1.0, x_star)))
    one_step_passed = (error < tol)

    return {
        "passed": one_step_passed,
        "x0": x0_val,
        "x_1": x_1,
        "x_star": x_star,
        "error_norm": error,
        "iterations": res.iterations,
        "result": res,
    }
