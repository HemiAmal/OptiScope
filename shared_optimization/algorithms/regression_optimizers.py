"""
Optimization routines for linear regression (Batch GD, SGD, Mini-batch SGD)
and binary logistic regression.
"""

import time
from typing import List, Optional, Tuple
from shared_optimization.core.vectors import norm_l2
from shared_optimization.core.convergence import OptimizationResult, IterationRecord, ConvergenceCriterion
from shared_optimization.problems.linear_regression import (
    LinearRegressionDataset,
    compute_mse_loss,
    compute_batch_gradient,
    compute_sample_gradient,
)
from shared_optimization.problems.logistic_regression import (
    LogisticRegressionDataset,
    compute_cross_entropy_loss,
    compute_logistic_gradient,
    classification_accuracy,
)


def train_linear_regression_gd(
    dataset: LinearRegressionDataset,
    learning_rate: float = 0.05,
    max_epochs: int = 200,
    tol: float = 1e-5,
) -> Tuple[List[float], float, OptimizationResult]:
    """
    Fits linear regression weights and bias using batch gradient descent.
    """
    start_time = time.time()
    w = [0.0] * dataset.d
    b = 0.0

    history: List[IterationRecord] = []
    loss_0 = compute_mse_loss(dataset, w, b)
    grad_w, grad_b = compute_batch_gradient(dataset, w, b)
    g_norm = norm_l2(grad_w + [grad_b])

    history.append(IterationRecord(iteration=0, x=w + [b], fun=loss_0, grad_norm=g_norm))
    converged = False

    for epoch in range(1, max_epochs + 1):
        if g_norm < tol:
            converged = True
            break

        # Batch update: w <- w - alpha * grad_w, b <- b - alpha * grad_b
        for j in range(dataset.d):
            w[j] -= learning_rate * grad_w[j]
        b -= learning_rate * grad_b

        loss = compute_mse_loss(dataset, w, b)
        grad_w, grad_b = compute_batch_gradient(dataset, w, b)
        g_norm = norm_l2(grad_w + [grad_b])

        history.append(
            IterationRecord(
                iteration=epoch,
                x=list(w) + [b],
                fun=loss,
                grad_norm=g_norm,
                step_size=learning_rate,
            )
        )

    exec_time = time.time() - start_time
    result = OptimizationResult(
        x=w + [b],
        fun=history[-1].fun,
        grad_norm=history[-1].grad_norm,
        iterations=len(history) - 1,
        func_evals=len(history),
        grad_evals=len(history),
        hess_evals=0,
        converged=converged,
        status_message="Converged" if converged else "Max epochs reached",
        history=history,
        execution_time=exec_time,
    )
    return w, b, result


def train_linear_regression_sgd(
    dataset: LinearRegressionDataset,
    learning_rate: float = 0.01,
    max_epochs: int = 50,
) -> Tuple[List[float], float, OptimizationResult]:
    """
    Fits linear regression via stochastic gradient descent (one sample per update).
    """
    start_time = time.time()
    w = [0.0] * dataset.d
    b = 0.0

    history: List[IterationRecord] = []
    loss_0 = compute_mse_loss(dataset, w, b)
    history.append(IterationRecord(iteration=0, x=w + [b], fun=loss_0, grad_norm=0.0))

    step_count = 0
    for epoch in range(1, max_epochs + 1):
        # Deterministic sample pass
        for i in range(dataset.n):
            grad_w, grad_b = compute_sample_gradient(dataset, i, w, b)
            for j in range(dataset.d):
                w[j] -= learning_rate * grad_w[j]
            b -= learning_rate * grad_b
            step_count += 1

        loss = compute_mse_loss(dataset, w, b)
        history.append(
            IterationRecord(
                iteration=epoch,
                x=list(w) + [b],
                fun=loss,
                grad_norm=0.0,
                step_size=learning_rate,
            )
        )

    exec_time = time.time() - start_time
    result = OptimizationResult(
        x=w + [b],
        fun=history[-1].fun,
        grad_norm=0.0,
        iterations=max_epochs,
        func_evals=max_epochs + 1,
        grad_evals=step_count,
        hess_evals=0,
        converged=True,
        status_message="SGD completed",
        history=history,
        execution_time=exec_time,
    )
    return w, b, result


def train_linear_regression_minibatch(
    dataset: LinearRegressionDataset,
    batch_size: int = 16,
    learning_rate: float = 0.02,
    max_epochs: int = 50,
) -> Tuple[List[float], float, OptimizationResult]:
    """
    Fits linear regression via mini-batch stochastic gradient descent.
    """
    start_time = time.time()
    w = [0.0] * dataset.d
    b = 0.0

    history: List[IterationRecord] = []
    loss_0 = compute_mse_loss(dataset, w, b)
    history.append(IterationRecord(iteration=0, x=w + [b], fun=loss_0, grad_norm=0.0))

    n = dataset.n
    d = dataset.d
    update_count = 0

    for epoch in range(1, max_epochs + 1):
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            m = end_idx - start_idx

            # Accumulate mini-batch gradient
            mb_grad_w = [0.0] * d
            mb_grad_b = 0.0
            for i in range(start_idx, end_idx):
                gw, gb = compute_sample_gradient(dataset, i, w, b)
                for j in range(d):
                    mb_grad_w[j] += gw[j]
                mb_grad_b += gb

            scale = 1.0 / m
            for j in range(d):
                w[j] -= learning_rate * (mb_grad_w[j] * scale)
            b -= learning_rate * (mb_grad_b * scale)
            update_count += 1

        loss = compute_mse_loss(dataset, w, b)
        history.append(
            IterationRecord(
                iteration=epoch,
                x=list(w) + [b],
                fun=loss,
                grad_norm=0.0,
                step_size=learning_rate,
            )
        )

    exec_time = time.time() - start_time
    result = OptimizationResult(
        x=w + [b],
        fun=history[-1].fun,
        grad_norm=0.0,
        iterations=max_epochs,
        func_evals=max_epochs + 1,
        grad_evals=update_count,
        hess_evals=0,
        converged=True,
        status_message="Mini-batch SGD completed",
        history=history,
        execution_time=exec_time,
    )
    return w, b, result


def train_logistic_regression(
    dataset: LogisticRegressionDataset,
    learning_rate: float = 0.1,
    max_epochs: int = 300,
    tol: float = 1e-4,
) -> Tuple[List[float], float, float, OptimizationResult]:
    """
    Fits binary logistic regression by minimizing binary cross-entropy loss.
    """
    start_time = time.time()
    w = [0.0] * dataset.d
    b = 0.0

    history: List[IterationRecord] = []
    loss_0 = compute_cross_entropy_loss(dataset, w, b)
    grad_w, grad_b = compute_logistic_gradient(dataset, w, b)
    g_norm = norm_l2(grad_w + [grad_b])

    history.append(IterationRecord(iteration=0, x=w + [b], fun=loss_0, grad_norm=g_norm))
    converged = False

    for epoch in range(1, max_epochs + 1):
        if g_norm < tol:
            converged = True
            break

        for j in range(dataset.d):
            w[j] -= learning_rate * grad_w[j]
        b -= learning_rate * grad_b

        loss = compute_cross_entropy_loss(dataset, w, b)
        grad_w, grad_b = compute_logistic_gradient(dataset, w, b)
        g_norm = norm_l2(grad_w + [grad_b])

        history.append(
            IterationRecord(
                iteration=epoch,
                x=list(w) + [b],
                fun=loss,
                grad_norm=g_norm,
                step_size=learning_rate,
            )
        )

    acc = classification_accuracy(dataset, w, b)
    exec_time = time.time() - start_time
    result = OptimizationResult(
        x=w + [b],
        fun=history[-1].fun,
        grad_norm=history[-1].grad_norm,
        iterations=len(history) - 1,
        func_evals=len(history),
        grad_evals=len(history),
        hess_evals=0,
        converged=converged,
        status_message="Converged" if converged else "Max epochs reached",
        history=history,
        execution_time=exec_time,
    )
    return w, b, acc, result
