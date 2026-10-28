"""
Logistic regression model, numerically stable sigmoid, BCE loss, and gradient computations:
    J(w, b) = - (1 / n) * sum_{i=1}^n [ y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) ]
"""

import math
from typing import List, Sequence, Tuple
from shared_optimization.core.vectors import validate_vector, dot_product, Number


def sigmoid(z: float) -> float:
    """Numerically stable sigmoid function."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


class LogisticRegressionDataset:
    """Stores feature matrix X and binary labels y in {0, 1}."""
    def __init__(self, X: Sequence[Sequence[Number]], y: Sequence[Number]) -> None:
        self.n = len(X)
        if self.n == 0:
            raise ValueError("Dataset cannot be empty.")
        if self.n != len(y):
            raise ValueError(f"Features count {self.n} != labels count {len(y)}.")

        self.X: List[List[float]] = [validate_vector(row, f"X[{i}]") for i, row in enumerate(X)]
        self.d = len(self.X[0])
        self.y: List[float] = [float(val) for val in y]
        for val in self.y:
            if val not in (0.0, 1.0):
                raise ValueError(f"Logistic regression target labels must be 0 or 1, got {val}.")


def generate_synthetic_classification_data(
    n_samples: int = 100,
    seed: int = 42,
) -> LogisticRegressionDataset:
    """
    Generate 2D synthetic linearly separable classification dataset.
    Class 0 clustered around (-1, -1), Class 1 clustered around (1, 1).
    """
    state = seed & 0xFFFFFFFF
    def pseudo_rand() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state / 0xFFFFFFFF) * 2.0 - 1.0

    X: List[List[float]] = []
    y: List[float] = []

    half = n_samples // 2
    # Class 0: Center (-1.2, -1.2)
    for _ in range(half):
        x1 = -1.2 + pseudo_rand() * 0.8
        x2 = -1.2 + pseudo_rand() * 0.8
        X.append([x1, x2])
        y.append(0.0)

    # Class 1: Center (+1.2, +1.2)
    for _ in range(n_samples - half):
        x1 = 1.2 + pseudo_rand() * 0.8
        x2 = 1.2 + pseudo_rand() * 0.8
        X.append([x1, x2])
        y.append(1.0)

    return LogisticRegressionDataset(X, y)


def compute_cross_entropy_loss(
    dataset: LogisticRegressionDataset, w: Sequence[Number], b: float, eps: float = 1e-15
) -> float:
    """Evaluate binary cross-entropy loss."""
    w_val = validate_vector(w, "w")
    total_loss = 0.0
    for i in range(dataset.n):
        z = dot_product(w_val, dataset.X[i]) + b
        p = sigmoid(z)
        p = max(eps, min(1.0 - eps, p))
        y_i = dataset.y[i]
        total_loss -= (y_i * math.log(p) + (1.0 - y_i) * math.log(1.0 - p))
    return total_loss / dataset.n


def compute_logistic_gradient(
    dataset: LogisticRegressionDataset, w: Sequence[Number], b: float
) -> Tuple[List[float], float]:
    """Compute analytical gradient of BCE loss: (grad_w, grad_b)."""
    w_val = validate_vector(w, "w")
    d = dataset.d
    n = dataset.n

    grad_w = [0.0] * d
    grad_b = 0.0

    for i in range(n):
        z = dot_product(w_val, dataset.X[i]) + b
        p = sigmoid(z)
        err = p - dataset.y[i]
        for j in range(d):
            grad_w[j] += err * dataset.X[i][j]
        grad_b += err

    scale = 1.0 / n
    for j in range(d):
        grad_w[j] *= scale
    grad_b *= scale

    return grad_w, grad_b


def predict_probabilities(
    dataset: LogisticRegressionDataset, w: Sequence[Number], b: float
) -> List[float]:
    """Return predicted probabilities p = sigma(w^T * x + b)."""
    w_val = validate_vector(w, "w")
    return [sigmoid(dot_product(w_val, x) + b) for x in dataset.X]


def classification_accuracy(
    dataset: LogisticRegressionDataset, w: Sequence[Number], b: float
) -> float:
    """Compute binary classification accuracy in [0, 1]."""
    probs = predict_probabilities(dataset, w, b)
    correct = 0
    for i, p in enumerate(probs):
        pred = 1.0 if p >= 0.5 else 0.0
        if pred == dataset.y[i]:
            correct += 1
    return correct / dataset.n
