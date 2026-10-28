"""
ResourceOpt experiment runner for constrained production allocation,
KKT optimality verification, and LICQ analysis.
"""

import sys
import os
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
LAB_ROOT = PROJECT_DIR.parent
for p in [str(PROJECT_DIR), str(LAB_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ResourceOpt.problems.production_allocation import ProductionAllocationProblem
from ResourceOpt.visualization.plots import (
    plot_production_polytope,
    plot_kkt_residuals_summary,
)


def run_resource_experiments() -> None:
    print("\n" + "=" * 70)
    print(" ResourceOpt: Production & Resource Allocation Under Constraints")
    print("=" * 70)
    
    results_dir = PROJECT_DIR / "results"
    plots_dir = results_dir / "plots"
    reports_dir = results_dir / "reports"
    for d in [plots_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    prob = ProductionAllocationProblem()
    
    print("\nRunning constrained allocation scenarios...")
    scenarios = prob.test_all_five_scenarios()
    
    hdr = f"{'Scenario Name':<42} | {'x Candidate':<14} | {'Stationarity':<12} | {'Primal Feas':<12} | {'LICQ':<6} | {'Verdict'}"
    print("\nConstrained KKT & LICQ Benchmark Summary:")
    print(hdr)
    print("-" * len(hdr))
    
    st_residuals = []
    pr_residuals = []
    scenario_labels = []
    
    for name, data in scenarios.items():
        if "kkt" in data:
            st = data["kkt"]["stationarity_residual"]
            pr = max(data["kkt"]["primal_eq_max_residual"], data["kkt"]["primal_ineq_max_violation"])
            licq_str = "True" if data["licq"]["licq_satisfied"] else "False"
            x_str = f"[{data['x'][0]:.2f}, {data['x'][1]:.2f}]"
            verdict = data["verdict"]
            print(f"{name:<42} | {x_str:<14} | {st:<12.5e} | {pr:<12.5e} | {licq_str:<6} | {verdict}")
            
            st_residuals.append(st)
            pr_residuals.append(pr)
            scenario_labels.append(name.split(":")[0])
        else:
            # LICQ focus cases
            x_str = f"[{data['x'][0]:.2f}, {data['x'][1]:.2f}]"
            licq_str = "True" if data["licq_satisfied"] else "False"
            det_val = data["gram_det"]
            print(f"{name:<42} | {x_str:<14} | GramDet={det_val:.2e} | ActCons={data['num_active']}     | {licq_str:<6} | {data['verdict']}")

    # Case 1 & Case 2 Coordinates for Plotting
    c1_x = scenarios["Case 1: Feasible Interior Optimum"]["x"]
    c2_x = scenarios["Case 2: Feasible Boundary Optimum"]["x"]
    
    polytope_plot_path = str(plots_dir / "production_polytope.png")
    kkt_bar_plot_path = str(plots_dir / "kkt_residuals_summary.png")
    
    plot_production_polytope(
        f_callable=prob.f,
        case1_opt=c1_x,
        case2_opt=c2_x,
        save_path=polytope_plot_path,
    )
    plot_kkt_residuals_summary(
        scenario_names=scenario_labels,
        stationarity_residuals=st_residuals,
        primal_residuals=pr_residuals,
        save_path=kkt_bar_plot_path,
    )
    
    print(f"\nSaved plots to:\n > {polytope_plot_path}\n > {kkt_bar_plot_path}")
    
    # Save summary report
    report_file = reports_dir / "resource_opt_summary.txt"
    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ResourceOpt Experimental Summary Report\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. Constrained Optimization Benchmark Results\n")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")
        for name, data in scenarios.items():
            if "kkt" in data:
                st = data["kkt"]["stationarity_residual"]
                pr = max(data["kkt"]["primal_eq_max_residual"], data["kkt"]["primal_ineq_max_violation"])
                licq_str = "True" if data["licq"]["licq_satisfied"] else "False"
                x_str = f"[{data['x'][0]:.2f}, {data['x'][1]:.2f}]"
                f.write(f"{name:<42} | {x_str:<14} | {st:<12.5e} | {pr:<12.5e} | {licq_str:<6} | {data['verdict']}\n")
            else:
                x_str = f"[{data['x'][0]:.2f}, {data['x'][1]:.2f}]"
                licq_str = "True" if data["licq_satisfied"] else "False"
                det_val = data["gram_det"]
                f.write(f"{name:<42} | {x_str:<14} | GramDet={det_val:.2e} | ActCons={data['num_active']}     | {licq_str:<6} | {data['verdict']}\n")
                
        f.write("\n2. Optimality and Constraint Qualification Checks\n")
        f.write(" - Lagrange multipliers solved via the direct saddle-point block KKT system [[Q, A^T], [A, 0]]\n")
        f.write(" - KKT verification evaluated stationarity, primal feasibility, dual feasibility, and complementary slackness\n")
        f.write(" - LICQ evaluated the determinant of the active constraint Jacobian Gram matrix\n")
        
    print(f"Summary report written to: {report_file}")
    print("=" * 78)


if __name__ == "__main__":
    run_resource_experiments()
