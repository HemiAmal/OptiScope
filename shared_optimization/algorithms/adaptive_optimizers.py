"""
Coordinate-wise adaptive learning rate optimizers: AdaGrad, RMSProp, and Adam.
"""

import time
from typing import Callable, List, Optional, Sequence
from shared_optimization.core.vectors import validate_vector, norm_l2, Number
from shared_optimization.core.convergence import (
    OptimizationResult,
    IterationRecord,
    ConvergenceCriterion,
    check_convergence,
    check_finite,
)

ScalarFunction = Callable[[Sequence[Number]], float]
GradientFunction = Callable[[Sequence[Number]], List[float]]


def adagrad_optimizer(
    f: ScalarFunction,
    grad_f: GradientFunction,
    x0: Sequence[Number],
    learning_rate: float = 0.1,
    epsilon: float = 1e-8,
    criterion: Optional[ConvergenceCriterion] = None,
) -> OptimizationResult:
    """
    Minimizes a scalar function using the AdaGrad adaptive optimizer.
    """
    start_time = time.time()
    crit = criterion if criterion is not None else ConvergenceCriterion()
    x = validate_vector(x0, "x0")
    check_finite(x, "x0")

    n = len(x)
    G = [0.0] * n  # Accumulated squared gradients

    history: List[IterationRecord] = []
    f_val = f(x)
    g = grad_f(x)
    g_norm = norm_l2(g)
    history.append(IterationRecord(iteration=0, x=list(x), fun=f_val, grad_norm=g_norm))

    converged = False
    status_msg = "Maximum iterations reached"

    for k in range(1, crit.max_iters + 1):
        if g_norm < crit.grad_tol:
            converged = True
            status_msg = f"Converged: gradient norm {g_norm:.2e} < {crit.grad_tol:.2e}"
            break

        # Accumulate squared gradients & step
        step = [0.0] * n
        for i in range(n):
            G[i] += g[i] * g[i]
            step[i] = - (learning_rate / (G[i] ** 0.5 + epsilon)) * g[i]

        x_next = [x[i] + step[i] for i in range(n)]
        check_finite(x_next, f"iterate_{k}")

        f_next = f(x_next)
        g_next = grad_f(x_next)
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
        func_evals=len(history),
        grad_evals=len(history),
        hess_evals=0,
        converged=converged,
        status_message=status_msg,
        history=history,
        execution_time=exec_time,
    )


def rmsprop_optimizer(
    f: ScalarFunction,
    grad_f: GradientFunction,
    x0: Sequence[Number],
    learning_rate: float = 0.01,
    beta: float = 0.9,
    epsilon: float = 1e-8,
    criterion: Optional[ConvergenceCriterion] = None,
) -> OptimizationResult:
    """
    Minimizes a scalar function using the RMSProp optimizer with moving average squared gradients.
    """
    start_time = time.time()
    crit = criterion if criterion is not None else ConvergenceCriterion()
    x = validate_vector(x0, "x0")
    check_finite(x, "x0")

    n = len(x)
    v = [0.0] * n  # Exponential moving average of squared gradients

    history: List[IterationRecord] = []
    f_val = f(x)
    g = grad_f(x)
    g_norm = norm_l2(g)
    history.append(IterationRecord(iteration=0, x=list(x), fun=f_val, grad_norm=g_norm))

    converged = False
    status_msg = "Maximum iterations reached"

    for k in range(1, crit.max_iters + 1):
        if g_norm < crit.grad_tol:
            converged = True
            status_msg = f"Converged: gradient norm {g_norm:.2e} < {crit.grad_tol:.2e}"
            break

        step = [0.0] * n
        for i in range(n):
            v[i] = beta * v[i] + (1.0 - beta) * (g[i] * g[i])
            step[i] = - (learning_rate / (v[i] ** 0.5 + epsilon)) * g[i]

        x_next = [x[i] + step[i] for i in range(n)]
        check_finite(x_next, f"iterate_{k}")

        f_next = f(x_next)
        g_next = grad_f(x_next)
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
        func_evals=len(history),
        grad_evals=len(history),
        hess_evals=0,
        converged=converged,
        status_message=status_msg,
        history=history,
        execution_time=exec_time,
    )


def adam_optimizer(
    f: ScalarFunction,
    grad_f: GradientFunction,
    x0: Sequence[Number],
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    criterion: Optional[ConvergenceCriterion] = None,
) -> OptimizationResult:
    """
    Minimizes a scalar function using the Adam optimizer with bias-corrected 1st and 2nd moments.
    """
    start_time = time.time()
    crit = criterion if criterion is not None else ConvergenceCriterion()
    x = validate_vector(x0, "x0")
    check_finite(x, "x0")

    n = len(x)
    m = [0.0] * n  # 1st moment vector
    v = [0.0] * n  # 2nd moment vector

    history: List[IterationRecord] = []
    f_val = f(x)
    g = grad_f(x)
    g_norm = norm_l2(g)
    history.append(IterationRecord(iteration=0, x=list(x), fun=f_val, grad_norm=g_norm))

    converged = False
    status_msg = "Maximum iterations reached"

    for k in range(1, crit.max_iters + 1):
        if g_norm < crit.grad_tol:
            converged = True
            status_msg = f"Converged: gradient norm {g_norm:.2e} < {crit.grad_tol:.2e}"
            break

        # Bias correction decay factors
        beta1_t = beta1 ** k
        beta2_t = beta2 ** k

        step = [0.0] * n
        for i in range(n):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            v[i] = beta2 * v[i] + (1.0 - beta2) * (g[i] * g[i])

            m_hat = m[i] / (1.0 - beta1_t)
            v_hat = v[i] / (1.0 - beta2_t)

            step[i] = - (learning_rate / (v_hat ** 0.5 + epsilon)) * m_hat

        x_next = [x[i] + step[i] for i in range(n)]
        check_finite(x_next, f"iterate_{k}")

        f_next = f(x_next)
        g_next = grad_f(x_next)
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
        func_evals=len(history),
        grad_evals=len(history),
        hess_evals=0,
        converged=converged,
        status_message=status_msg,
        history=history,
        execution_time=exec_time,
    )
