"""
Sparse predictive modeling via Lasso and subgradient optimization.

Formulation:
  J(w) = 1/(2n) ||Xw - y||^2 + lambda * ||w||_1

Subgradient:
  grad J(w) = 1/n X^T (Xw - y) + lambda * v
  where v_j = sign(w_j) if w_j != 0, else 0 in [-1, 1].
"""

import sys
import time
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared_optimization.problems.linear_regression import LinearRegressionDataset
from shared_optimization.algorithms.subgradient import (
    subgradient_method,
    subgradient_lasso,
)
from shared_optimization.problems.lasso import lasso_objective, lasso_subgradient


def run_general_subgradient_demo(
    dim: int = 5,
    x0: List[float] = None,
    max_iters: int = 250,
) -> Dict[str, Any]:
    """
    Subgradient method applied to the nonsmooth Euclidean norm f(x) = ||x||_2.
    Minimizer: x* = 0, f(x*) = 0.
    """
    if x0 is None:
        x0 = [3.0, -2.5, 4.0, -1.0, 2.0]
        
    def f_norm(x: List[float]) -> float:
        return float(np.linalg.norm(x))
        
    def subgrad_norm(x: List[float]) -> List[float]:
        xa = np.array(x, dtype=float)
        n = np.linalg.norm(xa)
        if n < 1e-12:
            return [0.0] * len(x)
        return (xa / n).tolist()
        
    t0 = time.perf_counter()
    res = subgradient_method(
        f=f_norm,
        subgrad_f=subgrad_norm,
        x0=x0,
        step_size_init=0.5,
        max_iters=max_iters,
        tol=1e-4,
    )
    elapsed = time.perf_counter() - t0
    
    return {
        "x0": x0,
        "final_x": res.x,
        "final_f": res.fun,
        "iterations": res.iterations,
        "history": [rec.fun for rec in res.history],
        "converged": res.converged,
        "time": elapsed,
    }


def run_lasso_regularization_path(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: List[float] = None,
    reporting_zero_threshold: float = 0.05,
    max_iters: int = 350,
) -> Dict[str, Any]:
    """
    Evaluates Lasso subgradient optimization across multiple regularization
    strengths lambda to observe feature shrinkage and parameter sparsity.
    """
    if lambdas is None:
        lambdas = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
        
    ds = LinearRegressionDataset(X.tolist(), y.tolist())
    
    path_results = []
    
    for lmb in lambdas:
        t0 = time.perf_counter()
        res = subgradient_lasso(
            dataset=ds,
            lmbda=lmb,
            step_size_init=0.08,
            max_iters=max_iters,
            tol=1e-4,
        )
        elapsed = time.perf_counter() - t0
        
        w_est = np.array(res.x)
        y_pred = X @ w_est
        mse = float(np.mean((y - y_pred) ** 2))
        
        # Count features below reporting threshold for sparsity analysis
        num_zeros = int(np.sum(np.abs(w_est) < reporting_zero_threshold))
        
        path_results.append({
            "lambda": lmb,
            "weights": res.x,
            "objective": res.fun,
            "mse": mse,
            "num_zeros": num_zeros,
            "iterations": res.iterations,
            "time": elapsed,
            "history": [rec.fun for rec in res.history],
            "converged": res.converged,
        })
        
    # Compute unregularized baseline (OLS solution: w = (X^T X)^(-1) X^T y)
    w_ols = np.linalg.pinv(X) @ y
    mse_ols = float(np.mean((y - X @ w_ols) ** 2))
    
    return {
        "lambdas": lambdas,
        "path": path_results,
        "ols_weights": w_ols.tolist(),
        "ols_mse": mse_ols,
        "reporting_threshold": reporting_zero_threshold,
    }
