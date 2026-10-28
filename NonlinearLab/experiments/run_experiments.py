"""
NonlinearLab experiment runner for quadratic benchmarks, Newton one-step demonstration,
and Rosenbrock calibration.
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

from NonlinearLab.problems.quadratic_benchmark import IllConditionedQuadratic
from NonlinearLab.problems.rosenbrock_calibration import RosenbrockCalibration
from NonlinearLab.visualization.plots import (
    plot_contour_trajectories,
    plot_convergence_history,
)


def run_nonlinear_experiments() -> None:
    print("\n" + "=" * 70)
    print(" NonlinearLab: Smooth Optimization Benchmarks")
    print("=" * 70)
    
    results_dir = PROJECT_DIR / "results"
    plots_dir = results_dir / "plots"
    hist_dir = results_dir / "histories"
    reports_dir = results_dir / "reports"
    for d in [plots_dir, hist_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Ill-conditioned quadratic benchmark
    print("\nRunning quadratic benchmark (kappa = 10.0)...")
    quad = IllConditionedQuadratic(cond_ratio=10.0)
    x0_quad = [-4.0, 3.0]
    
    quad_results = quad.run_all_quadratic_solvers(x0=x0_quad, max_iters=250)
    
    hdr = f"{'Algorithm':<22} | {'Final f(x)':<12} | {'||grad f||':<12} | {'Iters':<6} | {'Time (ms)':<9} | {'Status'}"
    print("\nQuadratic Optimization Results:")
    print(hdr)
    print("-" * len(hdr))
    
    quad_trajectories = {}
    quad_loss_hist = {}
    quad_grad_hist = {}
    
    for name, data in quad_results.items():
        res = data["result"]
        f_val = res.fun
        g_norm = res.grad_norm
        iters = res.iterations
        t_ms = res.execution_time * 1000.0
        status = "Converged" if res.converged else "MaxIters"
        print(f"{name:<22} | {f_val:<12.5e} | {g_norm:<12.5e} | {iters:<6} | {t_ms:<9.2f} | {status}")
        
        quad_trajectories[name] = [rec.x for rec in res.history]
        quad_loss_hist[name] = [rec.fun for rec in res.history]
        quad_grad_hist[name] = [rec.grad_norm for rec in res.history]
        
        fname = hist_dir / f"quad_{name.lower().replace(' ', '_')}.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "f_val", "grad_norm", "x1", "x2"])
            for rec in res.history:
                writer.writerow([rec.iteration, rec.fun, rec.grad_norm, rec.x[0], rec.x[1]])

    # Render quadratic trajectory and convergence plots
    quad_plot_path = str(plots_dir / "quadratic_trajectories.png")
    quad_conv_path = str(plots_dir / "quadratic_convergence.png")
    plot_contour_trajectories(
        f=quad.f,
        trajectories=quad_trajectories,
        x_range=(-5.0, 5.0),
        y_range=(-4.0, 4.0),
        title="Optimization Trajectories on Ill-Conditioned Quadratic",
        x_star=quad.x_star.tolist(),
        save_path=quad_plot_path,
    )
    plot_convergence_history(
        histories=quad_loss_hist,
        metric_name="Objective Gap $f(x_k) - f(x^*)$",
        title="Quadratic Objective Convergence",
        save_path=quad_conv_path,
    )

    # 2. Newton one-step convergence demonstration on quadratic
    print("\nNewton One-Step Convergence Demonstration:")
    x0_demo = [12.5, -8.0]
    one_step_res = quad.run_newton_one_step_demonstration(x0=x0_demo)
    print(f"  Initial Point x0:        {x0_demo}")
    print(f"  Exact Analytical x*:     {one_step_res['x_star']}")
    print(f"  Point after 1 Step x1:   {one_step_res['x_1']}")
    print(f"  Distance to Optimum:     {one_step_res['error_norm']:.6e}")
    print(f"  Convergence Verified:    {one_step_res['passed']}")

    # 3. Rosenbrock banana valley calibration
    print("\nRunning Rosenbrock valley benchmark...")
    rosen = RosenbrockCalibration()
    x0_rosen = [-1.2, 1.0]
    rosen_results = rosen.run_all_rosenbrock_solvers(x0=x0_rosen, max_iters=500)
    
    print("\n--- Rosenbrock Benchmark Results ---")
    print(hdr)
    print("-" * len(hdr))
    
    rosen_trajectories = {}
    rosen_loss_hist = {}
    
    for name, data in rosen_results.items():
        res = data["result"]
        f_val = res.fun
        g_norm = res.grad_norm
        iters = res.iterations
        t_ms = res.execution_time * 1000.0
        status = "Converged" if res.converged else "MaxIters"
        print(f"{name:<22} | {f_val:<12.5e} | {g_norm:<12.5e} | {iters:<6} | {t_ms:<9.2f} | {status}")
        
        rosen_trajectories[name] = [rec.x for rec in res.history]
        rosen_loss_hist[name] = [rec.fun for rec in res.history]
        
        fname = hist_dir / f"rosen_{name.lower().replace(' ', '_')}.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "f_val", "grad_norm", "x1", "x2"])
            for rec in res.history:
                writer.writerow([rec.iteration, rec.fun, rec.grad_norm, rec.x[0], rec.x[1]])

    # Render Rosenbrock plots
    rosen_plot_path = str(plots_dir / "rosenbrock_trajectories.png")
    rosen_conv_path = str(plots_dir / "rosenbrock_convergence.png")
    plot_contour_trajectories(
        f=rosen.f,
        trajectories=rosen_trajectories,
        x_range=(-2.0, 2.0),
        y_range=(-1.0, 3.0),
        title="Optimization Trajectories on Rosenbrock Valley",
        x_star=rosen.x_star,
        save_path=rosen_plot_path,
    )
    plot_convergence_history(
        histories=rosen_loss_hist,
        metric_name="Rosenbrock Objective $f(x)$",
        title="Rosenbrock Objective Convergence",
        save_path=rosen_conv_path,
    )

    print(f"\nPlots generated:\n  - {quad_plot_path}\n  - {quad_conv_path}\n  - {rosen_plot_path}\n  - {rosen_conv_path}")
    
    # Save text summary report
    report_file = reports_dir / "nonlinear_lab_summary.txt"
    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("NonlinearLab Experimental Summary Report\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. Ill-Conditioned Quadratic Benchmark (kappa=10.0)\n")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")
        for name, data in quad_results.items():
            res = data["result"]
            f.write(f"{name:<22} | {res.fun:<12.5e} | {res.grad_norm:<12.5e} | {res.iterations:<6} | {res.execution_time*1000:<9.2f} | {'Converged' if res.converged else 'MaxIters'}\n")
        f.write("\n2. Newton One-Step Demonstration\n")
        f.write(f"Initial: {x0_demo} -> 1-Step: {one_step_res['x_1']} (Exact: {one_step_res['x_star']})\n")
        f.write(f"Verified: {one_step_res['passed']} (Error Norm: {one_step_res['error_norm']:.6e})\n\n")
        f.write("3. Rosenbrock Valley Benchmark\n")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")
        for name, data in rosen_results.items():
            res = data["result"]
            f.write(f"{name:<22} | {res.fun:<12.5e} | {res.grad_norm:<12.5e} | {res.iterations:<6} | {res.execution_time*1000:<9.2f} | {'Converged' if res.converged else 'MaxIters'}\n")
            
    print(f"Summary report written to: {report_file}")
    print("=" * 78)


if __name__ == "__main__":
    run_nonlinear_experiments()
