"""
ResourceOpt Visualization Suite.
Generates 2D feasible region polytopes with constraint boundaries,
objective contours, and KKT residual diagnostic charts.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


def plot_production_polytope(
    f_callable,
    case1_opt: List[float],
    case2_opt: List[float],
    save_path: str,
) -> None:
    """
    Renders the 2D production optimization landscape, constraint boundaries,
    and verified optimal solutions.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    x1_vals = np.linspace(0.0, 8.0, 300)
    x2_vals = np.linspace(0.0, 10.0, 300)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    
    Z = np.zeros_like(X1)
    for i in range(300):
        for j in range(300):
            Z[i, j] = f_callable([X1[i, j], X2[i, j]])
            
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Objective contours
    cf = plt.contourf(X1, X2, Z, levels=30, cmap="Blues_r", alpha=0.6)
    plt.colorbar(cf, label="Production Cost $f(x_1, x_2)$")
    plt.contour(X1, X2, Z, levels=15, colors="grey", alpha=0.3, linewidths=0.6)
    
    # Constraint boundaries
    # 1. Target equality: x1 + x2 = 8.0 -> x2 = 8.0 - x1
    plt.plot(x1_vals, 8.0 - x1_vals, label="Demand Target: $x_1 + x_2 = 8.0$", color="#1f77b4", linewidth=2.5)
    
    # 2. Machine limit: 1.5 x1 + x2 = 10.0 -> x2 = 10.0 - 1.5 x1
    x2_machine = 10.0 - 1.5 * x1_vals
    plt.plot(x1_vals, x2_machine, label="Machine Hours: $1.5 x_1 + x_2 \\leq 10.0$", color="#d62728", linestyle="--", linewidth=2.5)
    
    # 3. Storage limit: x1 = 6.0
    plt.axvline(6.0, color="#2ca02c", linestyle=":", linewidth=2.0, label="Storage Limit: $x_1 \\leq 6.0$")
    
    # 4. Target equality for Case 2: x1 + x2 = 9.0
    plt.plot(x1_vals, 9.0 - x1_vals, label="Shifted Target: $x_1 + x_2 = 9.0$", color="#9467bd", linestyle="-.", linewidth=2.0)
    
    # Highlight Optima
    plt.plot(case1_opt[0], case1_opt[1], "go", markersize=10, markeredgecolor="black", label=f"Case 1: Interior Optimum ({case1_opt[0]:.1f}, {case1_opt[1]:.1f})")
    plt.plot(case2_opt[0], case2_opt[1], "r*", markersize=14, markeredgecolor="black", label=f"Case 2: Boundary Optimum ({case2_opt[0]:.1f}, {case2_opt[1]:.1f})")
    
    plt.title("ResourceOpt: 2D Production Constraints & Constrained Optima", fontsize=13, fontweight="bold")
    plt.xlabel("Product 1 Output $x_1$ (Units)", fontsize=11)
    plt.ylabel("Product 2 Output $x_2$ (Units)", fontsize=11)
    plt.xlim(0.0, 7.5)
    plt.ylim(0.0, 9.5)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_kkt_residuals_summary(
    scenario_names: List[str],
    stationarity_residuals: List[float],
    primal_residuals: List[float],
    save_path: str,
) -> None:
    """
    Bar chart comparing stationarity and primal feasibility residuals across tested cases.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    x = np.arange(len(scenario_names))
    width = 0.35
    
    # Clip residuals for log display
    st_clipped = [max(r, 1e-15) for r in stationarity_residuals]
    pr_clipped = [max(r, 1e-15) for r in primal_residuals]
    
    plt.figure(figsize=(10, 5), dpi=300)
    plt.bar(x - width/2, st_clipped, width, label="Stationarity Residual", color="#1f77b4")
    plt.bar(x + width/2, pr_clipped, width, label="Primal Violation Residual", color="#d62728")
    
    plt.yscale("log")
    plt.axhline(1e-4, color="black", linestyle="--", linewidth=1.0, label="Feasibility Tolerance ($10^{-4}$)")
    plt.title("ResourceOpt: KKT Condition Residual Comparison Across Scenarios", fontsize=13, fontweight="bold")
    plt.xticks(x, scenario_names, rotation=20, ha="right", fontsize=9)
    plt.ylabel("Residual Magnitude (Log Scale)", fontsize=11)
    plt.grid(True, which="both", axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
