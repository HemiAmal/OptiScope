import os
from typing import Dict, Any, List, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_contour_trajectories(
    f: Callable[[List[float]], float],
    trajectories: Dict[str, List[List[float]]],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title: str,
    x_star: List[float],
    save_path: str,
    num_grid: int = 250,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    xs = np.linspace(x_range[0], x_range[1], num_grid)
    ys = np.linspace(y_range[0], y_range[1], num_grid)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)
    for i in range(num_grid):
        for j in range(num_grid):
            Z[i, j] = f([X[i, j], Y[i, j]])

    plt.figure(figsize=(10, 8), dpi=300)

    levels = np.percentile(Z, np.linspace(0, 95, 30))
    cf = plt.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.8)
    plt.colorbar(cf, label="Objective Value $f(x)$")
    plt.contour(X, Y, Z, levels=levels, colors="white", alpha=0.25, linewidths=0.5)

    color_map = {
        "Fixed GD": "#1f77b4",
        "Exact Line Search GD": "#ff7f0e",
        "Armijo GD": "#2ca02c",
        "Polyak Momentum": "#9467bd",
        "Nesterov GD": "#8c564b",
        "Newton Method": "#e377c2",
        "Damped Newton": "#17becf",
        "BFGS": "#d62728",
    }

    for name, path in trajectories.items():
        if not path or len(path) == 0:
            continue
        p_arr = np.array(path)
        c = color_map.get(name, "#000000")
        plt.plot(p_arr[:, 0], p_arr[:, 1], marker="o", markersize=3,
                 label=name, color=c, linewidth=1.8, alpha=0.85)
        plt.plot(p_arr[0, 0], p_arr[0, 1], "ks", markersize=6)

    plt.plot(x_star[0], x_star[1], "r*", markersize=14, markeredgecolor="black", label="Optimum $x^*$")

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("$x_1$", fontsize=11)
    plt.ylabel("$x_2$", fontsize=11)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.legend(frameon=True, facecolor="white", loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_convergence_history(
    histories: Dict[str, List[float]],
    metric_name: str,
    title: str,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=300)
    for name, vals in histories.items():
        if vals and len(vals) > 0:
            plt.plot(range(len(vals)), vals, label=name, linewidth=1.8)

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Iteration $k$", fontsize=11)
    plt.ylabel(metric_name, fontsize=11)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
