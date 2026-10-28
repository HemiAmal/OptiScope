"""
Subgradient optimization methods for convex nonsmooth objectives and Lasso regularization.
"""

import time
from typing import Callable, List, Optional, Sequence
from shared_optimization.core.vectors import validate_vector, vec_add, scalar_mul, norm_l2, Number
from shared_optimization.core.convergence import OptimizationResult, IterationRecord, ConvergenceCriterion
from shared_optimization.problems.linear_regression import LinearRegressionDataset
from shared_optimization.problems.lasso import lasso_objective, lasso_subgradient

SubgradientFunction = Callable[[Sequence[Number]], List[float]]
ScalarFunction = Callable[[Sequence[Number]], float]


def subgradient_method(
    f: ScalarFunction,
    subgrad_f: SubgradientFunction,
    x0: Sequence[Number],
    step_size_init: float = 0.5,
    max_iters: int = 500,
    tol: float = 1e-4,
) -> OptimizationResult:
    """
    Minimizes a convex nonsmooth function using diminishing step sizes:
        alpha_k = step_size_init / sqrt(k)
        x_{k+1} = x_k - alpha_k * g_k,  where g_k in subdifferential(f)(x_k)
    """
    start_time = time.time()
    x = validate_vector(x0, "x0")
    n = len(x)

    history: List[IterationRecord] = []
    f_val = f(x)
    g = subgrad_f(x)
    g_norm = norm_l2(g)

    x_best = list(x)
    f_best = f_val

    history.append(IterationRecord(iteration=0, x=list(x), fun=f_val, grad_norm=g_norm))
    converged = False

    for k in range(1, max_iters + 1):
        if g_norm < tol:
            converged = True
            break

        # Diminishing square-summable step size: alpha_k = alpha_0 / sqrt(k)
        alpha_k = step_size_init / (k ** 0.5)

        # Update: x_{k+1} = x_k - alpha_k * g_k
        step = scalar_mul(-alpha_k, g)
        x_next = vec_add(x, step)

        f_next = f(x_next)
        g_next = subgrad_f(x_next)
        g_next_norm = norm_l2(g_next)

        # Track best point encountered
        if f_next < f_best:
            f_best = f_next
            x_best = list(x_next)

        history.append(
            IterationRecord(
                iteration=k,
                x=list(x_next),
                fun=f_next,
                grad_norm=g_next_norm,
                step_size=alpha_k,
                extra={"f_best": f_best},
            )
        )

        x, f_val, g, g_norm = x_next, f_next, g_next, g_next_norm

    exec_time = time.time() - start_time
    return OptimizationResult(
        x=x_best,
        fun=f_best,
        grad_norm=g_norm,
        iterations=len(history) - 1,
        func_evals=len(history),
        grad_evals=len(history),
        hess_evals=0,
        converged=converged,
        status_message="Converged" if converged else "Subgradient max iters reached",
        history=history,
        execution_time=exec_time,
    )


def subgradient_lasso(
    dataset: LinearRegressionDataset,
    lmbda: float = 0.1,
    step_size_init: float = 0.1,
    max_iters: int = 500,
    tol: float = 1e-4,
) -> OptimizationResult:
    """
    Minimizes the composite Lasso objective J(w) = (1 / (2n)) * ||Xw - y||_2^2 + lambda * ||w||_1
    using subgradient optimization.
    """
    start_time = time.time()
    w = [0.0] * dataset.d

    def obj_fn(v):
        return lasso_objective(dataset, v, lmbda)

    def subgrad_fn(v):
        return lasso_subgradient(dataset, v, lmbda, subgradient_at_zero=0.0)

    res = subgradient_method(
        f=obj_fn,
        subgrad_f=subgrad_fn,
        x0=w,
        step_size_init=step_size_init,
        max_iters=max_iters,
        tol=tol,
    )
    res.execution_time = time.time() - start_time
    return res
