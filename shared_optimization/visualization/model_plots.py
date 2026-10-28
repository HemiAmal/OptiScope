"""
Plotting utilities for regression fits, decision boundaries, and convergence comparison curves.
"""

import os
from typing import Dict, List, Optional, Sequence
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_optimization.core.convergence import OptimizationResult
from shared_optimization.problems.linear_regression import LinearRegressionDataset
from shared_optimization.problems.logistic_regression import LogisticRegressionDataset


def plot_linear_regression_fit(
    dataset: LinearRegressionDataset,
    w: float,
    b: float,
    filepath: str,
    title: str = "Linear Regression Model Fit",
) -> None:
    """Render 1D linear regression scatter points and learned regression line."""
    x_vals = np.array([row[0] for row in dataset.X])
    y_vals = np.array(dataset.y)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#181a1f")
    ax.set_facecolor("#21252b")

    # Data points
    ax.scatter(x_vals, y_vals, color="#61afef", alpha=0.8, edgecolors="#181a1f", s=40, label="Training Data (x, y)")

    # Fitted line
    x_line = np.linspace(np.min(x_vals) - 0.2, np.max(x_vals) + 0.2, 100)
    y_line = w * x_line + b
    ax.plot(x_line, y_line, color="#e5c07b", linewidth=2.5, label=f"Fit: y = {w:.2f}x + {b:.2f}")

    ax.set_title(title, color="#e5c07b", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Feature x", color="#abb2bf", fontsize=11)
    ax.set_ylabel("Target y", color="#abb2bf", fontsize=11)
    ax.tick_params(colors="#828997", which="both")
    for spine in ax.spines.values():
        spine.set_color("#3e4451")

    ax.legend(facecolor="#282c34", edgecolor="#3e4451", labelcolor="#abb2bf", fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor="none")
    if filepath.endswith(".svg"):
        plt.savefig(filepath[:-4] + ".png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def plot_logistic_regression_boundary(
    dataset: LogisticRegressionDataset,
    w: List[float],
    b: float,
    filepath: str,
    title: str = "Logistic Regression Decision Boundary",
) -> None:
    """Render 2D feature points and classification separating line: w1*x1 + w2*x2 + b = 0."""
    X = np.array(dataset.X)
    y = np.array(dataset.y)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#181a1f")
    ax.set_facecolor("#21252b")

    # Class 0 points
    ax.scatter(
        X[y == 0, 0], X[y == 0, 1],
        color="#e06c75", alpha=0.85, edgecolors="#181a1f", s=45, label="Class 0"
    )
    # Class 1 points
    ax.scatter(
        X[y == 1, 0], X[y == 1, 1],
        color="#98c379", alpha=0.85, edgecolors="#181a1f", s=45, label="Class 1"
    )

    # Decision boundary line: w1*x1 + w2*x2 + b = 0 => x2 = - (w1*x1 + b) / w2
    if abs(w[1]) > 1e-6:
        x1_span = np.linspace(np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5, 100)
        x2_boundary = - (w[0] * x1_span + b) / w[1]
        ax.plot(
            x1_span, x2_boundary,
            color="#e5c07b", linewidth=2.5, linestyle="--",
            label="Decision Boundary (p = 0.5)"
        )

    ax.set_title(title, color="#e5c07b", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("x1", color="#abb2bf", fontsize=11)
    ax.set_ylabel("x2", color="#abb2bf", fontsize=11)
    ax.tick_params(colors="#828997", which="both")
    for spine in ax.spines.values():
        spine.set_color("#3e4451")

    ax.legend(facecolor="#282c34", edgecolor="#3e4451", labelcolor="#abb2bf", fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor="none")
    if filepath.endswith(".svg"):
        plt.savefig(filepath[:-4] + ".png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def plot_convergence_curves(
    results_dict: Dict[str, OptimizationResult],
    filepath: str,
    title: str = "Optimization Convergence Comparison",
    log_scale: bool = True,
) -> None:
    """
    Plot objective value vs iteration across multiple algorithms.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("#181a1f")
    ax.set_facecolor("#21252b")

    colors = [
        "#61afef",  # Light blue
        "#98c379",  # Green
        "#e5c07b",  # Gold
        "#e06c75",  # Coral / Red
        "#c678dd",  # Purple
        "#56b6c2",  # Cyan
        "#d19a66",  # Orange
        "#e06cbb",  # Magenta
    ]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    for idx, (name, res) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]
        iters = [r.iteration for r in res.history]
        vals = [r.fun for r in res.history]

        if log_scale:
            # Shift positive to plot on log scale
            min_val = min(vals)
            shift = abs(min_val) + 1e-12 if min_val <= 0 else 0.0
            plot_vals = [v + shift for v in vals]
            ax.semilogy(iters, plot_vals, label=name, color=color, linestyle=ls, linewidth=2.0)
        else:
            ax.plot(iters, vals, label=name, color=color, linestyle=ls, linewidth=2.0)

    ax.set_title(title, color="#e5c07b", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Iteration Number k", color="#abb2bf", fontsize=11)
    ax.set_ylabel("Objective Value f(x_k)" + (" (log scale)" if log_scale else ""), color="#abb2bf", fontsize=11)
    ax.tick_params(colors="#828997", which="both")
    ax.grid(True, linestyle=":", color="#3e4451", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#3e4451")

    ax.legend(facecolor="#282c34", edgecolor="#3e4451", labelcolor="#abb2bf", fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor="none")
    if filepath.endswith(".svg"):
        plt.savefig(filepath[:-4] + ".png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
