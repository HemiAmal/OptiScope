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

from EnergyOpt.data.synthetic_energy import create_energy_regression_dataset
from EnergyOpt.problems.demand_regression import run_demand_prediction_benchmark
from EnergyOpt.problems.generation_dispatch import EconomicGenerationDispatch
from EnergyOpt.visualization.plots import (
    plot_demand_predictions,
    plot_regression_optimizer_convergence,
    plot_generation_dispatch_allocation,
)


def run_energy_experiments() -> None:
    print("\n" + "=" * 78)
    print(" PROJECT 1: ENERGYOPT — ELECTRICITY DEMAND & GENERATION PLANNING")
    print("=" * 78)

    results_dir = PROJECT_DIR / "results"
    plots_dir = results_dir / "plots"
    hist_dir = results_dir / "histories"
    reports_dir = results_dir / "reports"
    for d in [plots_dir, hist_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\nLoading synthetic hourly electricity demand dataset...")
    train_ds, val_ds = create_energy_regression_dataset(num_days=60, train_ratio=0.8, seed=42)
    print(f" > Training samples:   {len(train_ds.targets)} hours")
    print(f" > Validation samples: {len(val_ds.targets)} hours")
    print(f" > Features:           {', '.join(train_ds.feature_names)}")

    print("\nTraining linear regression models...")
    results = run_demand_prediction_benchmark(
        train_features=train_ds.features,
        train_targets=train_ds.targets,
        val_features=val_ds.features,
        val_targets=val_ds.targets,
    )

    for name, data in results.items():
        fname = hist_dir / f"energy_reg_{name.lower().replace(' ', '_').replace('-', '_')}.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "loss"])
            for i, val in enumerate(data["history"]):
                writer.writerow([i, val])

    print("\n--- Demand Prediction Performance Benchmark ---")
    hdr = f"{'Algorithm':<18} | {'Train RMSE':<12} | {'Val RMSE':<12} | {'Val R^2':<10} | {'Iters':<8} | {'Time (s)':<10}"
    print(hdr)
    print("-" * len(hdr))

    best_model_name = None
    best_val_rmse = float("inf")

    for name, data in results.items():
        tr_rmse = data["train_metrics"]["rmse"]
        val_rmse = data["val_metrics"]["rmse"]
        val_r2 = data["val_metrics"]["r2"]
        iters = data["iterations"]
        t = data["time"]
        print(f"{name:<18} | {tr_rmse:<12.3f} | {val_rmse:<12.3f} | {val_r2:<10.4f} | {iters:<8} | {t:<10.4f}")
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_name = name

    best_weights = results[best_model_name]["weights"]
    best_bias = results[best_model_name]["bias"]
    val_pred = val_ds.features @ np.array(best_weights) + best_bias

    pred_plot_path = str(plots_dir / "demand_predictions.png")
    conv_plot_path = str(plots_dir / "regression_loss_curves.png")
    plot_demand_predictions(val_ds.timestamps, val_ds.targets, val_pred, pred_plot_path)
    plot_regression_optimizer_convergence(results, conv_plot_path)
    print(f"\nSaved plots to:\n > {pred_plot_path}\n > {conv_plot_path}")

    print("\n" + "-" * 78)
    print(" Continuous Generation Resource Allocation via KKT & Lagrange Multipliers")
    print("-" * 78)

    peak_val_idx = int(np.argmax(val_pred[:24]))
    nominal_predicted_demand = float(val_pred[peak_val_idx])
    actual_demand = float(val_ds.targets[peak_val_idx])
    print(f" > Selected Dispatch Window: Hour {peak_val_idx}")
    print(f" > Model-Predicted Demand:  {nominal_predicted_demand:.2f} MW (via {best_model_name})")
    print(f" > Actual Realized Demand:  {actual_demand:.2f} MW")

    dispatch = EconomicGenerationDispatch(predicted_demand=nominal_predicted_demand)

    print("\nSolving Equality-Constrained Economic Dispatch...")
    lagrange_sol = dispatch.solve_equality_dispatch()
    p_opt = lagrange_sol["x"]
    lambda_opt = lagrange_sol["lambda"][0] if isinstance(lagrange_sol["lambda"], list) else lagrange_sol["lambda"]
    print(f" > Optimal Generation Allocation: P1 = {p_opt[0]:.2f} MW, P2 = {p_opt[1]:.2f} MW, P3 = {p_opt[2]:.2f} MW")
    print(f" > Total Generation:             {sum(p_opt):.2f} MW (Target: {nominal_predicted_demand:.2f} MW)")
    print(f" > System Marginal Price lambda:  ${lambda_opt:.2f}/MWh")
    print(f" > Total Production Cost:         ${dispatch.cost_function(p_opt):.2f}/h")

    print("\nEvaluating Karush-Kuhn-Tucker (KKT) Conditions...")
    kkt_eval = dispatch.evaluate_kkt_and_licq(P_candidate=p_opt, lambda_eq=lambda_opt)
    kkt_res = kkt_eval["kkt"]
    print(f" > Stationarity Residual:          {kkt_res['stationarity_residual']:.6e} (Passed: {kkt_res['stationarity_passed']})")
    print(f" > Primal Eq Max Residual:         {kkt_res['primal_eq_max_residual']:.6e} (Passed: {kkt_res['primal_feasibility_passed']})")
    print(f" > Dual Feasibility Passed:        {kkt_res['dual_feasibility_passed']}")
    print(f" > Complementary Slackness Max:    {kkt_res['complementary_slackness_max']:.6e}")
    print(f" > Overall Verified KKT Point:     {kkt_res['kkt_satisfied']}")

    print("\nChecking Linear Independence Constraint Qualification (LICQ)...")
    licq_res = kkt_eval["licq"]
    print(f" > Number of Active Constraints:   {licq_res['num_active_constraints']}")
    print(f" > Gram Determinant:               {licq_res['gram_determinant']:.6e}")
    print(f" > LICQ Satisfied:                 {licq_res['licq_satisfied']}")

    dispatch_plot_path = str(plots_dir / "generation_dispatch.png")
    plot_generation_dispatch_allocation(
        generators=["Gen 1 (Base Coal/Nuc)", "Gen 2 (CCGT Gas)", "Gen 3 (Peaker Gas)"],
        allocations=p_opt,
        capacities_min=dispatch.p_min.tolist(),
        capacities_max=dispatch.p_max.tolist(),
        total_demand=nominal_predicted_demand,
        save_path=dispatch_plot_path,
    )
    print(f"\nSaved dispatch allocation plot to: {dispatch_plot_path}")

    impact = dispatch.analyze_prediction_error_impact(true_demand=actual_demand, predicted_demand=nominal_predicted_demand)
    print("\n--- Operational Impact of Prediction Error ---")
    print(f" > Predicted Demand:   {impact['predicted_demand']:.2f} MW")
    print(f" > True Demand:        {impact['true_demand']:.2f} MW")
    print(f" > Forecast Error:     {impact['demand_error']:+.2f} MW")
    print(f" > Realized Shortfall: {impact['shortfall_mw']:.2f} MW")
    print(f" > Planned Base Cost:  ${impact['planned_cost']:.2f}")
    print(f" > Ideal Total Cost:   ${impact['ideal_cost']:.2f}")
    print(f" > Excess Cost Incurred: ${impact['excess_cost']:.2f}")

    report_file = reports_dir / "energy_opt_summary.txt"
    with open(report_file, "w") as f:
        f.write("=" * 75 + "\n")
        f.write("ENERGYOPT: EXPERIMENTAL SUMMARY REPORT\n")
        f.write("=" * 75 + "\n\n")
        f.write("1. DEMAND PREDICTION BENCHMARK\n")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")
        for name, data in results.items():
            f.write(f"{name:<18} | {data['train_metrics']['rmse']:<12.3f} | {data['val_metrics']['rmse']:<12.3f} | {data['val_metrics']['r2']:<10.4f} | {data['iterations']:<8} | {data['time']:<10.4f}\n")
        f.write(f"\nBest Predictive Model: {best_model_name} (Val RMSE: {best_val_rmse:.3f} MW)\n\n")
        f.write("2. ECONOMIC GENERATION DISPATCH & CONSTRAINED OPTIMIZATION\n")
        f.write(f"Target Demand: {nominal_predicted_demand:.2f} MW\n")
        f.write(f"Dispatched Generation: P1={p_opt[0]:.2f} MW, P2={p_opt[1]:.2f} MW, P3={p_opt[2]:.2f} MW\n")
        f.write(f"System Marginal Price (Lagrange Multiplier lambda): ${lambda_opt:.2f}/MWh\n")
        f.write(f"KKT Stationarity Residual: {kkt_res['stationarity_residual']:.6e} (Satisfied: {kkt_res['stationarity_passed']})\n")
        f.write(f"KKT Primal Feasibility: {kkt_res['primal_eq_max_residual']:.6e} (Satisfied: {kkt_res['primal_feasibility_passed']})\n")
        f.write(f"LICQ Active Constraints: {licq_res['num_active_constraints']}, Gram Det: {licq_res['gram_determinant']:.6e} (LICQ: {licq_res['licq_satisfied']})\n")
        f.write(f"\n3. FORECAST ERROR SENSITIVITY\n")
        f.write(f"Forecast Error: {impact['demand_error']:+.2f} MW -> Excess Cost Incurred: ${impact['excess_cost']:.2f}\n")

    print(f"Summary report saved to: {report_file}")
    print("=" * 78)


if __name__ == "__main__":
    run_energy_experiments()
