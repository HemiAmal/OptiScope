"""
SparseModel Visualization Suite.
Generates publication-quality figures for Lasso regularization paths,
coefficient recovery, and nonsmooth subgradient convergence.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


def plot_lasso_regularization_path(
    lambdas: List[float],
    weights_path: List[List[float]],
    feature_names: List[str],
    active_indices: List[int],
    save_path: str,
) -> None:
    """
    Plots the shrinkage of feature weights as lambda increases.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    W = np.array(weights_path)  # Shape (num_lambdas, p)
    p = len(feature_names)
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    for j in range(p):
        is_active = j in active_indices
        lw = 2.4 if is_active else 1.2
        ls = "-" if is_active else "--"
        alpha = 0.95 if is_active else 0.55
        lbl = f"{feature_names[j]} (Active)" if is_active else f"{feature_names[j]} (Irrelevant)"
        plt.plot(lambdas, W[:, j], label=lbl, linewidth=lw, linestyle=ls, alpha=alpha, marker="o", markersize=4)
        
    plt.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    plt.xscale("log")
    plt.title("SparseModel: Lasso Regularization Path (Subgradient Shrinkage)", fontsize=13, fontweight="bold")
    plt.xlabel("Regularization Strength $\\lambda$ (log scale)", fontsize=11)
    plt.ylabel("Estimated Weight Value $\\hat{w}_j$", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_coefficient_comparison(
    feature_names: List[str],
    true_weights: List[float],
    estimated_weights: List[float],
    ols_weights: List[float],
    save_path: str,
) -> None:
    """
    Side-by-side bar chart comparing True weights, OLS unregularized weights, and Lasso weights.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    p = len(feature_names)
    x = np.arange(p)
    width = 0.25
    
    plt.figure(figsize=(12, 6), dpi=300)
    plt.bar(x - width, true_weights, width, label="True Ground Truth $w^*$", color="#2ca02c", alpha=0.85)
    plt.bar(x, ols_weights, width, label="Unregularized OLS", color="#ff7f0e", alpha=0.75)
    plt.bar(x + width, estimated_weights, width, label="Lasso Subgradient", color="#1f77b4", alpha=0.9)
    
    plt.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    plt.title("SparseModel: True vs OLS vs Lasso Feature Weight Recovery", fontsize=13, fontweight="bold")
    plt.xticks(x, feature_names, rotation=35, ha="right", fontsize=10)
    plt.ylabel("Coefficient Magnitude", fontsize=11)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_subgradient_convergence(
    general_subgrad_hist: List[float],
    lasso_hist: List[float],
    save_path: str,
) -> None:
    """
    Plots convergence of subgradient optimization on nonsmooth test objectives.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # General Subgradient: f(x) = ||x||_2
    ax1.plot(range(len(general_subgrad_hist)), general_subgrad_hist, color="#d62728", linewidth=1.8)
    ax1.set_title("General Subgradient Method ($f(x) = ||x||_2$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Iteration", fontsize=10)
    ax1.set_ylabel("Norm $||x_k||_2$", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Lasso Subgradient Objective
    ax2.plot(range(len(lasso_hist)), lasso_hist, color="#1f77b4", linewidth=1.8)
    ax2.set_title("Lasso Subgradient Objective $J(w)$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Iteration", fontsize=10)
    ax2.set_ylabel("Lasso Objective $J(w_k)$", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("SparseModel: Nonsmooth Optimization Convergence Histories", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
