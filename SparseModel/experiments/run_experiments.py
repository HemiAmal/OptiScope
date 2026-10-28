"""
SparseModel experiment runner for subgradient optimization on nonsmooth norm
and Lasso regularized regression.
"""

import sys
import os
import csv
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
LAB_ROOT = PROJECT_DIR.parent
for p in [str(PROJECT_DIR), str(LAB_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from SparseModel.data.synthetic_sparse import create_sparse_regression_dataset
from SparseModel.problems.lasso_formulation import (
    run_general_subgradient_demo,
    run_lasso_regularization_path,
)
from SparseModel.visualization.plots import (
    plot_lasso_regularization_path,
    plot_coefficient_comparison,
    plot_subgradient_convergence,
)


def run_sparse_experiments() -> None:
    print("\n" + "=" * 70)
    print(" SparseModel: Subgradient Optimization & Lasso Benchmarks")
    print("=" * 70)
    
    results_dir = PROJECT_DIR / "results"
    plots_dir = results_dir / "plots"
    hist_dir = results_dir / "histories"
    reports_dir = results_dir / "reports"
    for d in [plots_dir, hist_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. General subgradient method on nonsmooth norm objective
    print("\nRunning subgradient descent on f(x) = ||x||_2...")
    subgrad_demo = run_general_subgradient_demo(dim=5, x0=[3.0, -2.5, 4.0, -1.0, 2.0], max_iters=250)
    init_val = subgrad_demo['final_f'] if len(subgrad_demo['history']) == 0 else subgrad_demo['history'][0]
    print(f"  Initial Norm ||x0||:   {init_val:.4f}")
    print(f"  Final Norm ||x*||:     {subgrad_demo['final_f']:.6e}")
    print(f"  Iterations Executed:   {subgrad_demo['iterations']}")
    print(f"  Runtime:               {subgrad_demo['time']*1000:.2f} ms")
    
    fname_15 = hist_dir / "general_subgradient_norm.csv"
    with open(fname_15, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "norm"])
        for i, val in enumerate(subgrad_demo["history"]):
            writer.writerow([i, val])

    # 2. Lasso subgradient optimization across regularization path
    print("\nEvaluating Lasso subgradient optimization over regularization path...")
    
    train_ds, val_ds = create_sparse_regression_dataset(num_samples=500, train_ratio=0.8, seed=42)
    print(f" > Samples:            {len(train_ds.targets)} train, {len(val_ds.targets)} validation")
    print(f" > Total Features:     {len(train_ds.feature_names)} candidate features")
    print(f" > Active Features:    Indices {train_ds.active_indices} (True weights: {train_ds.true_weights})")
    
    lambdas = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
    path_data = run_lasso_regularization_path(
        X=train_ds.features,
        y=train_ds.targets,
        lambdas=lambdas,
        reporting_zero_threshold=0.08,
        max_iters=350,
    )
    
    print("\n--- Lasso Regularization Path Benchmark ---")
    hdr = f"{'Lambda':<8} | {'Objective':<12} | {'Train MSE':<10} | {'Zero Coeffs':<12} | {'Iters':<6} | {'Time (s)':<8}"
    print(hdr)
    print("-" * len(hdr))
    
    weights_path = []
    for item in path_data["path"]:
        lmb = item["lambda"]
        obj = item["objective"]
        mse = item["mse"]
        zeros = item["num_zeros"]
        iters = item["iterations"]
        t = item["time"]
        weights_path.append(item["weights"])
        print(f"{lmb:<8.3f} | {obj:<12.5f} | {mse:<10.5f} | {zeros:<12d} / 8 | {iters:<6} | {t:<8.4f}")
        
        # Save CSV
        fname_lasso = hist_dir / f"lasso_subgrad_lambda_{lmb}.csv"
        with open(fname_lasso, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "loss"])
            for i, val in enumerate(item["history"]):
                writer.writerow([i, val])

    # Unregularized OLS comparison
    print(f"\nUnregularized OLS Baseline MSE: {path_data['ols_mse']:.5f}")
    
    # Select intermediate lambda for detailed coefficient comparison (e.g., lambda = 0.05)
    best_idx = 2  # lambda = 0.05
    best_item = path_data["path"][best_idx]
    best_w = best_item["weights"]
    print(f"\nRecovered Weights (lambda = {best_item['lambda']}):")
    for name, w_t, w_est, w_o in zip(train_ds.feature_names, train_ds.true_weights, best_w, path_data["ols_weights"]):
        print(f" > {name:<25}: True={w_t:+6.2f} | Lasso={w_est:+6.3f} | OLS={w_o:+6.3f}")

    # Generate Visualizations
    reg_plot_path = str(plots_dir / "lasso_regularization_path.png")
    bar_plot_path = str(plots_dir / "coefficient_comparison.png")
    conv_plot_path = str(plots_dir / "subgradient_convergence.png")
    
    plot_lasso_regularization_path(
        lambdas=lambdas,
        weights_path=weights_path,
        feature_names=train_ds.feature_names,
        active_indices=train_ds.active_indices,
        save_path=reg_plot_path,
    )
    plot_coefficient_comparison(
        feature_names=train_ds.feature_names,
        true_weights=train_ds.true_weights.tolist(),
        estimated_weights=best_w,
        ols_weights=path_data["ols_weights"],
        save_path=bar_plot_path,
    )
    plot_subgradient_convergence(
        general_subgrad_hist=subgrad_demo["history"],
        lasso_hist=best_item["history"],
        save_path=conv_plot_path,
    )
    
    print(f"\nSaved plots to:\n > {reg_plot_path}\n > {bar_plot_path}\n > {conv_plot_path}")
    
    # Save summary report
    report_file = reports_dir / "sparse_model_summary.txt"
    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SparseModel Experimental Summary Report\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. Nonsmooth Norm Minimization: f(x) = ||x||_2\n")
        f.write(f"Initial: {subgrad_demo['x0']}\n")
        f.write(f"Final Minimum Norm: {subgrad_demo['final_f']:.6e} in {subgrad_demo['iterations']} iters\n\n")
        f.write("2. Lasso Regularization Path (Sparsity Recovery)\n")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")
        for item in path_data["path"]:
            f.write(f"{item['lambda']:<8.3f} | {item['objective']:<12.5f} | {item['mse']:<10.5f} | {item['num_zeros']:<12d} / 8 | {item['iterations']:<6} | {item['time']:<8.4f}\n")
        f.write(f"\nOLS Baseline MSE: {path_data['ols_mse']:.5f}\n\n")
        f.write("3. Coefficient Recovery Breakdown (lambda = 0.050):\n")
        for name, w_t, w_est, w_o in zip(train_ds.feature_names, train_ds.true_weights, best_w, path_data["ols_weights"]):
            f.write(f" - {name:<25}: True={w_t:+6.2f} | Lasso={w_est:+6.3f} | OLS={w_o:+6.3f}\n")
            
    print(f"Summary report written to: {report_file}")
    print("=" * 78)


if __name__ == "__main__":
    run_sparse_experiments()
