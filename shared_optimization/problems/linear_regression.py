"""
Linear regression dataset container, prediction, and MSE loss calculations:
    J(w, b) = (1 / n) * sum_{i=1}^n (y_hat_i - y_i)^2
"""

from typing import List, Sequence, Tuple
from shared_optimization.core.vectors import validate_vector, dot_product, Number


class LinearRegressionDataset:
    """Stores feature matrix X and target vector y for linear regression."""
    def __init__(self, X: Sequence[Sequence[Number]], y: Sequence[Number]) -> None:
        self.n = len(X)
        if self.n == 0:
            raise ValueError("Dataset cannot be empty.")
        if self.n != len(y):
            raise ValueError(f"Features count {self.n} != targets count {len(y)}.")

        self.X: List[List[float]] = [validate_vector(row, f"X[{i}]") for i, row in enumerate(X)]
        self.d = len(self.X[0])
        self.y: List[float] = validate_vector(y, "y")


def generate_synthetic_linear_data(
    n_samples: int = 100,
    true_w: Sequence[Number] = (2.5, -1.5),
    true_b: float = 4.0,
    noise_amplitude: float = 0.05,
    seed: int = 42,
) -> LinearRegressionDataset:
    """
    Generate synthetic data with deterministic pseudo-random generator (no external random library).
    Using a standard Linear Congruential Generator (LCG) for full reproducibility.
    """
    w = validate_vector(true_w, "true_w")
    d = len(w)

    # Simple deterministic LCG pseudo-random generator
    state = seed & 0xFFFFFFFF
    def pseudo_rand() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state / 0xFFFFFFFF) * 2.0 - 1.0  # Uniform in [-1, 1]

    X: List[List[float]] = []
    y: List[float] = []

    for _ in range(n_samples):
        row = [pseudo_rand() * 2.0 for _ in range(d)]
        # True linear combination: y = w^T * x + b + noise
        val = true_b
        for j in range(d):
            val += w[j] * row[j]
        val += pseudo_rand() * noise_amplitude
        X.append(row)
        y.append(val)

    return LinearRegressionDataset(X, y)


def compute_mse_loss(
    dataset: LinearRegressionDataset, w: Sequence[Number], b: float
) -> float:
    """Evaluate MSE loss on dataset."""
    w_val = validate_vector(w, "w")
    total_sq_err = 0.0
    for i in range(dataset.n):
        y_hat = dot_product(w_val, dataset.X[i]) + b
        diff = y_hat - dataset.y[i]
        total_sq_err += diff * diff
    return total_sq_err / dataset.n


def compute_batch_gradient(
    dataset: LinearRegressionDataset, w: Sequence[Number], b: float
) -> Tuple[List[float], float]:
    """Compute analytical gradient over full dataset: (grad_w, grad_b)."""
    w_val = validate_vector(w, "w")
    d = dataset.d
    n = dataset.n

    grad_w = [0.0] * d
    grad_b = 0.0

    for i in range(n):
        y_hat = dot_product(w_val, dataset.X[i]) + b
        diff = y_hat - dataset.y[i]
        for j in range(d):
            grad_w[j] += diff * dataset.X[i][j]
        grad_b += diff

    scale = 2.0 / n
    for j in range(d):
        grad_w[j] *= scale
    grad_b *= scale

    return grad_w, grad_b


def compute_sample_gradient(
    dataset: LinearRegressionDataset, sample_idx: int, w: Sequence[Number], b: float
) -> Tuple[List[float], float]:
    """Compute stochastic gradient for a single sample: (grad_w, grad_b)."""
    w_val = validate_vector(w, "w")
    d = dataset.d
    x_i = dataset.X[sample_idx]
    y_i = dataset.y[sample_idx]

    y_hat = dot_product(w_val, x_i) + b
    diff = 2.0 * (y_hat - y_i)

    grad_w = [diff * x_i[j] for j in range(d)]
    grad_b = diff
    return grad_w, grad_b
