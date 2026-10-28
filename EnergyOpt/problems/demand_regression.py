import sys
import time
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared_optimization.problems.linear_regression import LinearRegressionDataset
from shared_optimization.algorithms.regression_optimizers import (
    train_linear_regression_gd,
    train_linear_regression_sgd,
    train_linear_regression_minibatch,
)
from shared_optimization.algorithms.adaptive_optimizers import (
    adagrad_optimizer,
    rmsprop_optimizer,
    adam_optimizer,
)
from shared_optimization.core.convergence import ConvergenceCriterion


def evaluate_regression_metrics(
    X: np.ndarray,
    y: np.ndarray,
    weights: List[float],
    bias: float,
) -> Dict[str, float]:
    w = np.array(weights)
    y_pred = X @ w + bias
    residuals = y - y_pred
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    y_var = float(np.var(y))
    r2 = 1.0 - (mse / y_var) if y_var > 1e-12 else 0.0
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def train_adaptive_linear_regression(
    dataset: LinearRegressionDataset,
    method: str = "adam",
    learning_rate: float = 0.05,
    max_epochs: int = 150,
) -> Tuple[List[float], float, Any, List[float]]:
    X = np.array(dataset.X)
    y = np.array(dataset.y)
    n, d = X.shape

    X_aug = np.column_stack([X, np.ones(n)])

    def loss_func(theta: List[float]) -> float:
        th = np.array(theta)
        err = X_aug @ th - y
        return float(np.mean(err ** 2) / 2.0)

    def grad_func(theta: List[float]) -> List[float]:
        th = np.array(theta)
        grad = (X_aug.T @ (X_aug @ th - y)) / n
        return grad.tolist()

    theta_0 = [0.0] * d + [float(np.mean(y))]
    criterion = ConvergenceCriterion(max_iters=max_epochs, grad_tol=1e-5, step_tol=1e-6)

    start_time = time.perf_counter()
    if method.lower() == "adagrad":
        result = adagrad_optimizer(loss_func, grad_func, theta_0, learning_rate=learning_rate, criterion=criterion)
    elif method.lower() == "rmsprop":
        result = rmsprop_optimizer(loss_func, grad_func, theta_0, learning_rate=learning_rate, criterion=criterion)
    elif method.lower() == "adam":
        result = adam_optimizer(loss_func, grad_func, theta_0, learning_rate=learning_rate, criterion=criterion)
    else:
        raise ValueError(f"Unknown optimizer: {method}")

    best_theta = result.x
    weights = list(best_theta[:d])
    bias = float(best_theta[d])
    loss_history = [rec.fun for rec in result.history]

    return weights, bias, result, loss_history


def run_demand_prediction_benchmark(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
) -> Dict[str, Any]:
    train_ds = LinearRegressionDataset(train_features.tolist(), train_targets.tolist())
    val_X = np.array(val_features)
    val_y = np.array(val_targets)

    results = {}

    t0 = time.perf_counter()
    w_gd, b_gd, res_gd = train_linear_regression_gd(train_ds, learning_rate=0.08, max_epochs=200, tol=1e-5)
    t_gd = time.perf_counter() - t0
    results["Batch GD"] = {
        "alg_num": 8,
        "weights": w_gd,
        "bias": b_gd,
        "train_metrics": evaluate_regression_metrics(train_features, train_targets, w_gd, b_gd),
        "val_metrics": evaluate_regression_metrics(val_X, val_y, w_gd, b_gd),
        "iterations": res_gd.iterations,
        "time": t_gd,
        "history": [rec.fun for rec in res_gd.history],
        "converged": res_gd.converged,
    }

    t0 = time.perf_counter()
    w_sgd, b_sgd, res_sgd = train_linear_regression_sgd(train_ds, learning_rate=0.01, max_epochs=60)
    t_sgd = time.perf_counter() - t0
    results["SGD"] = {
        "alg_num": 9,
        "weights": w_sgd,
        "bias": b_sgd,
        "train_metrics": evaluate_regression_metrics(train_features, train_targets, w_sgd, b_sgd),
        "val_metrics": evaluate_regression_metrics(val_X, val_y, w_sgd, b_sgd),
        "iterations": res_sgd.iterations,
        "time": t_sgd,
        "history": [rec.fun for rec in res_sgd.history],
        "converged": res_sgd.converged,
    }

    t0 = time.perf_counter()
    w_mb, b_mb, res_mb = train_linear_regression_minibatch(train_ds, batch_size=32, learning_rate=0.03, max_epochs=60)
    t_mb = time.perf_counter() - t0
    results["Mini-batch SGD"] = {
        "alg_num": 10,
        "weights": w_mb,
        "bias": b_mb,
        "train_metrics": evaluate_regression_metrics(train_features, train_targets, w_mb, b_mb),
        "val_metrics": evaluate_regression_metrics(val_X, val_y, w_mb, b_mb),
        "iterations": res_mb.iterations,
        "time": t_mb,
        "history": [rec.fun for rec in res_mb.history],
        "converged": res_mb.converged,
    }

    t0 = time.perf_counter()
    w_ada, b_ada, res_ada, hist_ada = train_adaptive_linear_regression(train_ds, method="adagrad", learning_rate=1.5, max_epochs=200)
    t_ada = time.perf_counter() - t0
    results["AdaGrad"] = {
        "alg_num": 12,
        "weights": w_ada,
        "bias": b_ada,
        "train_metrics": evaluate_regression_metrics(train_features, train_targets, w_ada, b_ada),
        "val_metrics": evaluate_regression_metrics(val_X, val_y, w_ada, b_ada),
        "iterations": res_ada.iterations,
        "time": t_ada,
        "history": hist_ada,
        "converged": res_ada.converged,
    }

    t0 = time.perf_counter()
    w_rms, b_rms, res_rms, hist_rms = train_adaptive_linear_regression(train_ds, method="rmsprop", learning_rate=0.1, max_epochs=200)
    t_rms = time.perf_counter() - t0
    results["RMSProp"] = {
        "alg_num": 13,
        "weights": w_rms,
        "bias": b_rms,
        "train_metrics": evaluate_regression_metrics(train_features, train_targets, w_rms, b_rms),
        "val_metrics": evaluate_regression_metrics(val_X, val_y, w_rms, b_rms),
        "iterations": res_rms.iterations,
        "time": t_rms,
        "history": hist_rms,
        "converged": res_rms.converged,
    }

    t0 = time.perf_counter()
    w_adam, b_adam, res_adam, hist_adam = train_adaptive_linear_regression(train_ds, method="adam", learning_rate=0.2, max_epochs=200)
    t_adam = time.perf_counter() - t0
    results["Adam"] = {
        "alg_num": 14,
        "weights": w_adam,
        "bias": b_adam,
        "train_metrics": evaluate_regression_metrics(train_features, train_targets, w_adam, b_adam),
        "val_metrics": evaluate_regression_metrics(val_X, val_y, w_adam, b_adam),
        "iterations": res_adam.iterations,
        "time": t_adam,
        "history": hist_adam,
        "converged": res_adam.converged,
    }

    return results
