import sys
import time
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared_optimization.problems.logistic_regression import LogisticRegressionDataset
from shared_optimization.algorithms.regression_optimizers import train_logistic_regression
from shared_optimization.algorithms.adaptive_optimizers import (
    adagrad_optimizer,
    rmsprop_optimizer,
    adam_optimizer,
)
from shared_optimization.core.convergence import ConvergenceCriterion


def sigmoid(z: np.ndarray) -> np.ndarray:
    z_clipped = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    eps = 1e-12
    p_safe = np.clip(y_prob, eps, 1.0 - eps)
    bce = float(-np.mean(y_true * np.log(p_safe) + (1 - y_true) * np.log(1.0 - p_safe)))

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "bce": bce,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": threshold,
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = None,
) -> Dict[str, List[float]]:
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.1, 0.9, 17)]

    accs, precs, recs, f1s = [], [], [], []
    for t in thresholds:
        m = compute_classification_metrics(y_true, y_prob, threshold=t)
        accs.append(m["accuracy"])
        precs.append(m["precision"])
        recs.append(m["recall"])
        f1s.append(m["f1"])

    return {
        "thresholds": thresholds,
        "accuracies": accs,
        "precisions": precs,
        "recalls": recs,
        "f1_scores": f1s,
    }


def train_adaptive_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "adam",
    learning_rate: float = 0.05,
    max_epochs: int = 200,
) -> Tuple[List[float], float, Any, List[float]]:
    n, d = X.shape
    X_aug = np.column_stack([X, np.ones(n)])
    y_arr = np.array(y)
    eps = 1e-12

    def bce_loss(theta: List[float]) -> float:
        th = np.array(theta)
        p = sigmoid(X_aug @ th)
        p_safe = np.clip(p, eps, 1.0 - eps)
        return float(-np.mean(y_arr * np.log(p_safe) + (1.0 - y_arr) * np.log(1.0 - p_safe)))

    def bce_grad(theta: List[float]) -> List[float]:
        th = np.array(theta)
        p = sigmoid(X_aug @ th)
        grad = (X_aug.T @ (p - y_arr)) / n
        return grad.tolist()

    pos_rate = max(min(float(np.mean(y_arr)), 0.95), 0.05)
    init_bias = float(np.log(pos_rate / (1.0 - pos_rate)))
    theta_0 = [0.0] * d + [init_bias]

    criterion = ConvergenceCriterion(max_iters=max_epochs, grad_tol=1e-5, step_tol=1e-6)

    start_time = time.perf_counter()
    if method.lower() == "adagrad":
        result = adagrad_optimizer(bce_loss, bce_grad, theta_0, learning_rate=learning_rate, criterion=criterion)
    elif method.lower() == "rmsprop":
        result = rmsprop_optimizer(bce_loss, bce_grad, theta_0, learning_rate=learning_rate, criterion=criterion)
    elif method.lower() == "adam":
        result = adam_optimizer(bce_loss, bce_grad, theta_0, learning_rate=learning_rate, criterion=criterion)
    else:
        raise ValueError(f"Unknown optimizer: {method}")

    best_theta = result.x
    weights = list(best_theta[:d])
    bias = float(best_theta[d])
    loss_history = [rec.fun for rec in result.history]

    return weights, bias, result, loss_history


def run_predictive_maintenance_benchmark(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
) -> Dict[str, Any]:
    train_ds = LogisticRegressionDataset(train_features.tolist(), train_labels.tolist())
    val_X = np.array(val_features)
    val_y = np.array(val_labels)

    results = {}

    t0 = time.perf_counter()
    w_log, b_log, acc_log, res_log = train_logistic_regression(
        train_ds, learning_rate=0.2, max_epochs=200, tol=1e-4
    )
    t_log = time.perf_counter() - t0
    val_prob_log = sigmoid(val_X @ np.array(w_log) + b_log)
    train_prob_log = sigmoid(train_features @ np.array(w_log) + b_log)

    results["Logistic Reg (GD)"] = {
        "alg_num": 11,
        "weights": w_log,
        "bias": b_log,
        "train_metrics": compute_classification_metrics(train_labels, train_prob_log),
        "val_metrics": compute_classification_metrics(val_y, val_prob_log),
        "val_prob": val_prob_log,
        "iterations": res_log.iterations,
        "time": t_log,
        "history": [rec.fun for rec in res_log.history],
        "converged": res_log.converged,
    }

    t0 = time.perf_counter()
    w_ada, b_ada, res_ada, hist_ada = train_adaptive_logistic_regression(
        train_features, train_labels, method="adagrad", learning_rate=0.8, max_epochs=200
    )
    t_ada = time.perf_counter() - t0
    val_prob_ada = sigmoid(val_X @ np.array(w_ada) + b_ada)
    train_prob_ada = sigmoid(train_features @ np.array(w_ada) + b_ada)
    results["AdaGrad"] = {
        "alg_num": 12,
        "weights": w_ada,
        "bias": b_ada,
        "train_metrics": compute_classification_metrics(train_labels, train_prob_ada),
        "val_metrics": compute_classification_metrics(val_y, val_prob_ada),
        "val_prob": val_prob_ada,
        "iterations": res_ada.iterations,
        "time": t_ada,
        "history": hist_ada,
        "converged": res_ada.converged,
    }

    t0 = time.perf_counter()
    w_rms, b_rms, res_rms, hist_rms = train_adaptive_logistic_regression(
        train_features, train_labels, method="rmsprop", learning_rate=0.04, max_epochs=200
    )
    t_rms = time.perf_counter() - t0
    val_prob_rms = sigmoid(val_X @ np.array(w_rms) + b_rms)
    train_prob_rms = sigmoid(train_features @ np.array(w_rms) + b_rms)
    results["RMSProp"] = {
        "alg_num": 13,
        "weights": w_rms,
        "bias": b_rms,
        "train_metrics": compute_classification_metrics(train_labels, train_prob_rms),
        "val_metrics": compute_classification_metrics(val_y, val_prob_rms),
        "val_prob": val_prob_rms,
        "iterations": res_rms.iterations,
        "time": t_rms,
        "history": hist_rms,
        "converged": res_rms.converged,
    }

    t0 = time.perf_counter()
    w_adam, b_adam, res_adam, hist_adam = train_adaptive_logistic_regression(
        train_features, train_labels, method="adam", learning_rate=0.05, max_epochs=200
    )
    t_adam = time.perf_counter() - t0
    val_prob_adam = sigmoid(val_X @ np.array(w_adam) + b_adam)
    train_prob_adam = sigmoid(train_features @ np.array(w_adam) + b_adam)
    results["Adam"] = {
        "alg_num": 14,
        "weights": w_adam,
        "bias": b_adam,
        "train_metrics": compute_classification_metrics(train_labels, train_prob_adam),
        "val_metrics": compute_classification_metrics(val_y, val_prob_adam),
        "val_prob": val_prob_adam,
        "iterations": res_adam.iterations,
        "time": t_adam,
        "history": hist_adam,
        "converged": res_adam.converged,
    }

    return results
