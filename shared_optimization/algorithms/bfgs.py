"""
Broyden-Fletcher-Goldfarb-Shanno (BFGS) Quasi-Newton optimizer.
Maintains a rank-2 update approximation of the inverse Hessian matrix.
"""

import time
from typing import Callable, List, Optional, Sequence
from shared_optimization.core.vectors import validate_vector, vec_add, vec_sub, scalar_mul, dot_product, norm_l2, Number
from shared_optimization.core.matrices import identity_matrix, mat_vec_mul, mat_mat_mul, mat_add, mat_sub, mat_scalar_mul, Matrix
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


def outer_product(u: List[float], v: List[float]) -> Matrix:
    """Computes vector outer product M = u * v^T where M_{ij} = u_i * v_j."""
    m = len(u)
    n = len(v)
    M: Matrix = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            M[i][j] = u[i] * v[j]
    return M


def bfgs_optimizer(
    f: ScalarFunction,
    grad_f: GradientFunction,
    x0: Sequence[Number],
    alpha_init: float = 1.0,
    rho_bt: float = 0.5,
    c1: float = 1e-4,
    criterion: Optional[ConvergenceCriterion] = None,
) -> OptimizationResult:
    """
    Minimizes a scalar function using the BFGS Quasi-Newton algorithm with Armijo line search.
    """
    start_time = time.time()
    crit = criterion if criterion is not None else ConvergenceCriterion()
    x = validate_vector(x0, "x0")
    check_finite(x, "x0")

    n = len(x)
    # Initial inverse Hessian approximation H_0 = I_n
    H = identity_matrix(n)
    I = identity_matrix(n)

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

        # Search direction: p = - H * g
        Hg = mat_vec_mul(H, g)
        p = scalar_mul(-1.0, Hg)

        # Check descent direction: <g, p> < 0
        direct_deriv = dot_product(g, p)
        if direct_deriv >= 0:
            # If numerical error caused loss of descent, reset H to identity
            H = identity_matrix(n)
            p = scalar_mul(-1.0, g)

        # Backtracking line search with Armijo condition
        def armijo_check(trial_alpha: float) -> bool:
            return armijo_condition(f, x, p, trial_alpha, f_val, g, c1=c1)

        alpha, bt_evals = backtracking_line_search(
            f=f,
            x=x,
            p=p,
            acceptance_fn=armijo_check,
            alpha_init=alpha_init,
            rho=rho_bt,
        )
        func_evals += bt_evals

        # Apply update
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

        # Compute BFGS displacement vectors
        # s_k = x_{k+1} - x_k
        s = vec_sub(x_next, x)
        # y_k = g_{k+1} - g_k
        y = vec_sub(g_next, g)

        ys = dot_product(y, s)
        if ys > 1e-12:
            rho = 1.0 / ys

            # V = I - rho * (s * y^T)
            syT = outer_product(s, y)
            V = mat_sub(I, mat_scalar_mul(rho, syT))

            # V_T = I - rho * (y * s^T)
            ysT = outer_product(y, s)
            V_T = mat_sub(I, mat_scalar_mul(rho, ysT))

            # ssT = s * s^T
            ssT = outer_product(s, s)

            # H_{k+1} = V * H * V_T + rho * (s * s^T)
            VH = mat_mat_mul(V, H)
            VHV_T = mat_mat_mul(VH, V_T)
            H = mat_add(VHV_T, mat_scalar_mul(rho, ssT))

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
