import os
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


def plot_classification_loss_curves(
    results: Dict[str, Dict[str, Any]],
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=300)
    colors = {
        "Logistic Reg (GD)": "#1f77b4",
        "AdaGrad": "#8c564b",
        "RMSProp": "#e377c2",
        "Adam": "#d62728",
    }

    for name, data in results.items():
        hist = data["history"]
        if hist and len(hist) > 0:
            c = colors.get(name, "#333333")
            plt.plot(range(len(hist)), hist, label=f"{name}", color=c, linewidth=2.0)

    plt.title("Binary Cross-Entropy Loss Convergence", fontsize=13, fontweight="bold")
    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_decision_boundary_2d(
    X_raw: np.ndarray,
    y: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    weights: List[float],
    bias: float,
    feature_idx_x: int = 0,
    feature_idx_y: int = 1,
    feature_names: List[str] = None,
    save_path: str = "decision_boundary.png",
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fx_name = feature_names[feature_idx_x] if feature_names else f"Feature {feature_idx_x}"
    fy_name = feature_names[feature_idx_y] if feature_names else f"Feature {feature_idx_y}"

    x_min, x_max = X_raw[:, feature_idx_x].min() - 0.5, X_raw[:, feature_idx_x].max() + 0.5
    y_min, y_max = X_raw[:, feature_idx_y].min() - 3.0, X_raw[:, feature_idx_y].max() + 3.0

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )

    grid_x_norm = (grid_x - scaler_mean[feature_idx_x]) / scaler_std[feature_idx_x]
    grid_y_norm = (grid_y - scaler_mean[feature_idx_y]) / scaler_std[feature_idx_y]

    logits = weights[feature_idx_x] * grid_x_norm + weights[feature_idx_y] * grid_y_norm + bias
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))

    plt.figure(figsize=(10, 7), dpi=300)

    cf = plt.contourf(grid_x, grid_y, probs, levels=np.linspace(0, 1, 21), cmap="RdYlBu_r", alpha=0.7)
    cbar = plt.colorbar(cf)
    cbar.set_label("Predicted Failure Probability $P(y=1)$", fontsize=11)

    cs = plt.contour(grid_x, grid_y, probs, levels=[0.5], colors="black", linewidths=2.5, linestyles="--")
    plt.clabel(cs, fmt={0.5: "Threshold 0.50"}, fontsize=10)

    normal_mask = (y == 0)
    fail_mask = (y == 1)
    plt.scatter(X_raw[normal_mask, feature_idx_x], X_raw[normal_mask, feature_idx_y],
                color="#1f77b4", edgecolors="black", alpha=0.7, s=40, label="Normal (0)")
    plt.scatter(X_raw[fail_mask, feature_idx_x], X_raw[fail_mask, feature_idx_y],
                color="#d62728", edgecolors="black", alpha=0.9, s=55, marker="^", label="Failure (1)")

    plt.title(f"Decision Boundary ({fx_name} vs {fy_name})", fontsize=13, fontweight="bold")
    plt.xlabel(fx_name, fontsize=11)
    plt.ylabel(fy_name, fontsize=11)
    plt.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(
    cm: List[List[int]],
    model_name: str,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_arr = np.array(cm)

    plt.figure(figsize=(6, 5), dpi=300)
    plt.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold")
    plt.colorbar()

    classes = ["Normal (0)", "Failure (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm_arr[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm_arr[i, j] > thresh else "black",
                     fontsize=14, fontweight="bold")

    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_threshold_tradeoffs(
    sweep_data: Dict[str, List[float]],
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    t = sweep_data["thresholds"]
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(t, sweep_data["accuracies"], label="Accuracy", color="#1f77b4", linewidth=2.0)
    plt.plot(t, sweep_data["precisions"], label="Precision", color="#ff7f0e", linewidth=2.0)
    plt.plot(t, sweep_data["recalls"], label="Recall", color="#2ca02c", linewidth=2.0)
    plt.plot(t, sweep_data["f1_scores"], label="F1 Score", color="#d62728", linestyle="--", linewidth=2.5)

    plt.title("Classification Metric Sensitivity Across Thresholds", fontsize=13, fontweight="bold")
    plt.xlabel("Decision Threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="lower center", ncol=4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
