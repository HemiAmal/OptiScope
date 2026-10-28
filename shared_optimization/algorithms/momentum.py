"""
Polyak momentum gradient descent (heavy-ball method).

Update Equations:
    v_{k+1} = beta * v_k + alpha * grad f(x_k)
    x_{k+1} = x_k - v_{k+1}
"""

import time
from typing import Callable, List, Optional, Sequence
from shared_optimization.core.vectors import validate_vector, vec_add, scalar_mul, norm_l2, Number
from shared_optimization.core.convergence import (
    OptimizationResult,
    IterationRecord,
    ConvergenceCriterion,
    check_convergence,
    check_finite,
)

ScalarFunction = Callable[[Sequence[Number]], float]
GradientFunction = Callable[[Sequence[Number]], List[float]]


def momentum_gd(
    f: ScalarFunction,
    grad_f: GradientFunction,
    x0: Sequence[Number],
    learning_rate: float = 0.001,
    beta: float = 0.9,
    criterion: Optional[ConvergenceCriterion] = None,
) -> OptimizationResult:
    """
    Minimizes a scalar function using Polyak's heavy-ball momentum method.
    """
    start_time = time.time()
    crit = criterion if criterion is not None else ConvergenceCriterion()
    x = validate_vector(x0, "x0")
    check_finite(x, "x0")

    n = len(x)
    v = [0.0] * n  # Velocity vector initialized to 0

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

        # v_{k+1} = beta * v_k + alpha * g_k
        v_next = vec_add(scalar_mul(beta, v), scalar_mul(learning_rate, g))
        # x_{k+1} = x_k - v_{k+1}
        x_next = vec_add(x, scalar_mul(-1.0, v_next))
        check_finite(x_next, f"iterate_{k}")

        f_next = f(x_next)
        func_evals += 1
        g_next = grad_f(x_next)
        grad_evals += 1
        g_next_norm = norm_l2(g_next)
        step_norm = norm_l2(v_next)
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

        x, f_val, g, g_norm, v = x_next, f_next, g_next, g_next_norm, v_next

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
