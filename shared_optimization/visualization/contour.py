"""
Contour and trajectory visualization utilities for 2D optimization landscapes.
"""

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

from shared_optimization.core.vectors import Number


def plot_contour_with_trajectory(
    f: Callable[[Sequence[Number]], float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    trajectory: List[List[float]],
    filepath: str,
    title: str = "Optimization Trajectory",
    num_grid: int = 150,
    num_levels: int = 25,
    true_optimum: Optional[List[float]] = None,
) -> None:
    """
    Generate a high-resolution filled contour plot with an overlaid optimization path using Matplotlib.

    Parameters:
        f: 2D scalar objective function.
        x_range: (x_min, x_max) bounding box.
        y_range: (y_min, y_max) bounding box.
        trajectory: List of [x_k, y_k] iterates visited by optimizer.
        filepath: Destination path (e.g. .svg or .png).
        title: Plot title.
        num_grid: Grid resolution (default: 150x150).
        num_levels: Number of contour intervals.
        true_optimum: Optional known global minimizer [x*, y*].
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    x_vals = np.linspace(x_min, x_max, num_grid)
    y_vals = np.linspace(y_min, y_max, num_grid)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Evaluate objective over grid
    Z = np.zeros_like(X)
    for i in range(num_grid):
        for j in range(num_grid):
            Z[i, j] = f([float(X[i, j]), float(Y[i, j])])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor("#181a1f")
    ax.set_facecolor("#21252b")

    # Determine logarithmic or linear levels
    z_min, z_max = np.min(Z), np.max(Z)
    if z_min >= 0 and z_max > z_min:
        # Use log-spaced levels if min >= 0
        levels = np.logspace(np.log10(max(z_min, 1e-4)), np.log10(z_max), num_levels)
    else:
        levels = np.linspace(z_min, z_max, num_levels)

    # Filled contour & contour lines
    cp = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.85)
    cbar = fig.colorbar(cp, ax=ax, shrink=0.85)
    cbar.set_label("Objective Value f(x)", color="#abb2bf", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="#5c6370")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#abb2bf")

    ax.contour(X, Y, Z, levels=levels, colors="#3e4451", linewidths=0.5, alpha=0.6)

    # Plot optimization trajectory
    if len(trajectory) > 1:
        traj_x = [pt[0] for pt in trajectory]
        traj_y = [pt[1] for pt in trajectory]
        ax.plot(
            traj_x,
            traj_y,
            color="#e5c07b",
            linewidth=2.2,
            linestyle="-",
            marker="o",
            markersize=3.5,
            markerfacecolor="#d19a66",
            markeredgecolor="#181a1f",
            label="Optimization Path",
            zorder=4,
        )

        # Plot Start & Final points
        ax.scatter(
            [traj_x[0]],
            [traj_y[0]],
            color="#98c379",
            s=90,
            edgecolors="#181a1f",
            linewidth=1.5,
            zorder=5,
            label="Start Point x0",
        )
        ax.scatter(
            [traj_x[-1]],
            [traj_y[-1]],
            color="#e06c75",
            s=90,
            edgecolors="#181a1f",
            linewidth=1.5,
            zorder=5,
            label=f"Final Iterate ({len(trajectory)-1} iters)",
        )

    # Plot True Optimum if provided
    if true_optimum is not None:
        ax.scatter(
            [true_optimum[0]],
            [true_optimum[1]],
            color="#56b6c2",
            marker="*",
            s=160,
            edgecolors="#181a1f",
            linewidth=1.5,
            zorder=6,
            label="Optimum x*",
        )

    ax.set_title(title, color="#e5c07b", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("x1", color="#abb2bf", fontsize=11)
    ax.set_ylabel("x2", color="#abb2bf", fontsize=11)
    ax.tick_params(colors="#828997", which="both")
    for spine in ax.spines.values():
        spine.set_color("#3e4451")

    ax.legend(facecolor="#282c34", edgecolor="#3e4451", labelcolor="#abb2bf", fontsize=9, loc="upper right")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor="none")

    # Also save .png version if saving .svg
    if filepath.endswith(".svg"):
        png_path = filepath[:-4] + ".png"
        plt.savefig(png_path, facecolor=fig.get_facecolor(), edgecolor="none")

    plt.close(fig)


def plot_contour_with_multiple_trajectories(
    f: Callable[[Sequence[Number]], float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    trajectories_dict: Dict[str, List[List[float]]],
    filepath: str,
    title: str = "Optimization Trajectories Comparison",
    num_grid: int = 150,
    num_levels: int = 25,
    true_optimum: Optional[List[float]] = None,
) -> None:
    """
    Generate a high-resolution contour plot with multiple optimizer trajectories overlaid.

    Parameters:
        f: 2D scalar objective function.
        x_range: (x_min, x_max) bounding box.
        y_range: (y_min, y_max) bounding box.
        trajectories_dict: Dictionary mapping algorithm name to list of [x_k, y_k] iterates.
        filepath: Destination path (.svg / .png).
        title: Plot title.
        num_grid: Resolution grid size.
        num_levels: Number of contour levels.
        true_optimum: Optional known global minimizer [x*, y*].
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    x_vals = np.linspace(x_min, x_max, num_grid)
    y_vals = np.linspace(y_min, y_max, num_grid)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.zeros_like(X)
    for i in range(num_grid):
        for j in range(num_grid):
            Z[i, j] = f([float(X[i, j]), float(Y[i, j])])

    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=150)
    fig.patch.set_facecolor("#181a1f")
    ax.set_facecolor("#21252b")

    z_min, z_max = np.min(Z), np.max(Z)
    if z_min >= 0 and z_max > z_min:
        levels = np.logspace(np.log10(max(z_min, 1e-4)), np.log10(z_max), num_levels)
    else:
        levels = np.linspace(z_min, z_max, num_levels)

    cp = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.8)
    cbar = fig.colorbar(cp, ax=ax, shrink=0.85)
    cbar.set_label("Objective Value f(x, y)", color="#abb2bf", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="#5c6370")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#abb2bf")

    ax.contour(X, Y, Z, levels=levels, colors="#3e4451", linewidths=0.5, alpha=0.5)

    # Distinct palette and line styles for up to 10 algorithms
    palette = [
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
    markers = ["o", "s", "^", "v", "D", "P", "X", "*"]

    # Plot start point once if available
    first_traj = next(iter(trajectories_dict.values()), [])
    if len(first_traj) > 0:
        start_pt = first_traj[0]
        ax.scatter(
            [start_pt[0]],
            [start_pt[1]],
            color="#ffffff",
            marker="o",
            s=100,
            edgecolors="#181a1f",
            linewidth=2.0,
            zorder=7,
            label=f"Start Point x0 ({start_pt[0]:.1f}, {start_pt[1]:.1f})",
        )

    # Plot each trajectory
    for idx, (name, traj) in enumerate(trajectories_dict.items()):
        if not traj:
            continue
        color = palette[idx % len(palette)]
        ls = linestyles[idx % len(linestyles)]
        mk = markers[idx % len(markers)]

        traj_x = [pt[0] for pt in traj]
        traj_y = [pt[1] for pt in traj]

        # Draw trajectory line
        ax.plot(
            traj_x,
            traj_y,
            color=color,
            linewidth=1.8,
            linestyle=ls,
            marker=mk,
            markersize=3.5,
            markerfacecolor=color,
            markeredgecolor="#181a1f",
            alpha=0.9,
            label=f"{name} ({len(traj)-1} iters)",
            zorder=5,
        )

    # Plot True Optimum if provided
    if true_optimum is not None:
        ax.scatter(
            [true_optimum[0]],
            [true_optimum[1]],
            color="#ffd700",
            marker="*",
            s=220,
            edgecolors="#181a1f",
            linewidth=1.5,
            zorder=8,
            label=f"Optimum x* ({true_optimum[0]:.1f}, {true_optimum[1]:.1f})",
        )

    ax.set_title(title, color="#e5c07b", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("x", color="#abb2bf", fontsize=11)
    ax.set_ylabel("y", color="#abb2bf", fontsize=11)
    ax.tick_params(colors="#828997", which="both")
    for spine in ax.spines.values():
        spine.set_color("#3e4451")

    ax.legend(facecolor="#282c34", edgecolor="#3e4451", labelcolor="#abb2bf", fontsize=8.5, loc="upper right")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor="none")

    if filepath.endswith(".svg"):
        png_path = filepath[:-4] + ".png"
        plt.savefig(png_path, facecolor=fig.get_facecolor(), edgecolor="none")

    plt.close(fig)

