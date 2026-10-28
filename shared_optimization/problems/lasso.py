"""
Lasso objective function and subgradient formulation:
    J(w) = (1 / (2*n)) * ||X * w - y||_2^2 + lambda * ||w||_1

Subgradient of Absolute Value:
For scalar z:
    partial |z| = { +1 }        if z > 0
    partial |z| = { -1 }        if z < 0
    partial |z| = [-1, +1]      if z = 0 (subdifferential interval)

Subgradient of Lasso Objective:
    g_w in (1 / n) * X^T * (X * w - y) + lambda * partial ||w||_1
"""

from typing import List, Sequence, Tuple
from shared_optimization.core.vectors import validate_vector, dot_product, norm_l1, Number
from shared_optimization.problems.linear_regression import LinearRegressionDataset


def subgradient_abs(val: float, subgradient_at_zero: float = 0.0) -> float:
    """
    Compute a subgradient of f(z) = |z|.

    Mathematical Definition:
        +1.0 if z > 0
        -1.0 if z < 0
        subgradient_at_zero in [-1.0, 1.0] if z == 0
    """
    if val > 1e-12:
        return 1.0
    elif val < -1e-12:
        return -1.0
    else:
        # Clamped to [-1.0, 1.0]
        return max(-1.0, min(1.0, subgradient_at_zero))


def lasso_objective(
    dataset: LinearRegressionDataset, w: Sequence[Number], lmbda: float
) -> float:
    """
    Evaluate Lasso objective J(w) = (1 / (2n)) * ||X*w - y||_2^2 + lambda * ||w||_1.
    """
    w_val = validate_vector(w, "w")
    total_sq_res = 0.0
    for i in range(dataset.n):
        y_hat = dot_product(w_val, dataset.X[i])
        res = y_hat - dataset.y[i]
        total_sq_res += res * res

    mse_half = total_sq_res / (2.0 * dataset.n)
    l1_pen = lmbda * norm_l1(w_val)
    return mse_half + l1_pen


def lasso_subgradient(
    dataset: LinearRegressionDataset,
    w: Sequence[Number],
    lmbda: float,
    subgradient_at_zero: float = 0.0,
) -> List[float]:
    """
    Compute a subgradient vector g in partial J(w).
    """
    w_val = validate_vector(w, "w")
    d = dataset.d
    n = dataset.n

    # Least squares gradient component: (1 / n) * X^T * (X*w - y)
    g = [0.0] * d
    for i in range(n):
        y_hat = dot_product(w_val, dataset.X[i])
        res = y_hat - dataset.y[i]
        for j in range(d):
            g[j] += res * dataset.X[i][j]

    for j in range(d):
        g[j] /= n
        # Add lambda * subgradient(|w_j|)
        g[j] += lmbda * subgradient_abs(w_val[j], subgradient_at_zero=subgradient_at_zero)

    return g
