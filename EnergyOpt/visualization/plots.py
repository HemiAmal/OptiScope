import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


def plot_demand_predictions(
    timestamps: np.ndarray,
    actual_demands: np.ndarray,
    predicted_demands: np.ndarray,
    save_path: str,
    num_samples_to_show: int = 120,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = min(len(actual_demands), num_samples_to_show)
    y_act = actual_demands[:n]
    y_pred = predicted_demands[:n]

    plt.figure(figsize=(12, 5), dpi=300)
    plt.plot(range(n), y_act, label="Actual Demand (MW)", color="#1f77b4", linewidth=2.0)
    plt.plot(range(n), y_pred, label="Predicted Demand (MW)", color="#ff7f0e", linestyle="--", linewidth=2.0)

    plt.title("Hourly Electricity Demand Prediction (Validation Window)", fontsize=13, fontweight="bold")
    plt.xlabel("Time (Hours)", fontsize=11)
    plt.ylabel("Electricity Demand (MW)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_regression_optimizer_convergence(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=300)

    colors = {
        "Batch GD": "#1f77b4",
        "SGD": "#2ca02c",
        "Mini-batch SGD": "#9467bd",
        "AdaGrad": "#8c564b",
        "RMSProp": "#e377c2",
        "Adam": "#d62728",
    }

    for name, data in results.items():
        hist = data["history"]
        if hist and len(hist) > 0:
            c = colors.get(name, "#333333")
            plt.plot(range(len(hist)), hist, label=f"{name}", color=c, linewidth=1.8)

    plt.title("Linear Regression Loss Convergence Across Optimizers", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch / Iteration", fontsize=11)
    plt.ylabel("MSE Loss (MW²)", fontsize=11)
    plt.yscale("log")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_generation_dispatch_allocation(
    generators: List[str],
    allocations: List[float],
    capacities_min: List[float],
    capacities_max: List[float],
    total_demand: float,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x_pos = np.arange(len(generators))
    width = 0.4

    plt.figure(figsize=(9, 5), dpi=300)
    plt.bar(x_pos, allocations, width, label="Dispatched Power (MW)", color="#2b5c8f")

    for i in range(len(generators)):
        plt.hlines(capacities_max[i], x_pos[i] - 0.25, x_pos[i] + 0.25, colors="#d9534f", linestyles="--", linewidth=2.5, label="Max Capacity" if i==0 else "")
        plt.hlines(capacities_min[i], x_pos[i] - 0.25, x_pos[i] + 0.25, colors="#5cb85c", linestyles=":", linewidth=2.5, label="Min Stable Limit" if i==0 else "")
        plt.text(x_pos[i], allocations[i] + 4.0, f"{allocations[i]:.1f} MW", ha="center", fontweight="bold", fontsize=10)

    plt.title(f"Economic Generation Dispatch (Total Demand = {total_demand:.1f} MW)", fontsize=13, fontweight="bold")
    plt.xticks(x_pos, generators, fontsize=10)
    plt.ylabel("Generation Output (MW)", fontsize=11)
    plt.ylim(0, max(capacities_max) * 1.15)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
