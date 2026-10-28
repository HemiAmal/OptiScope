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

from PredictMaint.data.synthetic_sensors import create_predictive_maintenance_dataset
from PredictMaint.problems.failure_classification import (
    run_predictive_maintenance_benchmark,
    sweep_thresholds,
)
from PredictMaint.visualization.plots import (
    plot_classification_loss_curves,
    plot_decision_boundary_2d,
    plot_confusion_matrix,
    plot_threshold_tradeoffs,
)


def run_maintenance_experiments() -> None:
    print("\n" + "=" * 78)
    print(" PROJECT 2: PREDICTMAINT — INDUSTRIAL PREDICTIVE MAINTENANCE CLASSIFIER")
    print("=" * 78)

    results_dir = PROJECT_DIR / "results"
    plots_dir = results_dir / "plots"
    hist_dir = results_dir / "histories"
    reports_dir = results_dir / "reports"
    for d in [plots_dir, hist_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\nGenerating synthetic industrial machinery telemetry dataset...")
    train_ds, val_ds = create_predictive_maintenance_dataset(num_samples=1200, train_ratio=0.8, seed=42)
    print(f" > Training records:   {len(train_ds.labels)} units (Failure rate: {np.mean(train_ds.labels):.1%})")
    print(f" > Validation records: {len(val_ds.labels)} units (Failure rate: {np.mean(val_ds.labels):.1%})")
    print(f" > Sensor channels:    {', '.join(train_ds.feature_names)}")

    print("\nTraining failure classifiers...")
    results = run_predictive_maintenance_benchmark(
        train_features=train_ds.features,
        train_labels=train_ds.labels,
        val_features=val_ds.features,
        val_labels=val_ds.labels,
    )

    for name, data in results.items():
        fname = hist_dir / f"maint_opt_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "loss"])
            for i, val in enumerate(data["history"]):
                writer.writerow([i, val])

    print("\n--- Failure Risk Classification Benchmark ---")
    hdr = f"{'Algorithm':<20} | {'Val BCE':<10} | {'Accuracy':<10} | {'Precision':<11} | {'Recall':<10} | {'F1':<8} | {'Time (s)':<10}"
    print(hdr)
    print("-" * len(hdr))

    best_model_name = None
    best_val_f1 = -1.0

    for name, data in results.items():
        vm = data["val_metrics"]
        bce = vm["bce"]
        acc = vm["accuracy"]
        prec = vm["precision"]
        rec = vm["recall"]
        f1 = vm["f1"]
        t = data["time"]
        print(f"{name:<20} | {bce:<10.4f} | {acc:<10.1%} | {prec:<11.1%} | {rec:<10.1%} | {f1:<8.3f} | {t:<10.4f}")
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model_name = name

    best_data = results[best_model_name]
    best_weights = best_data["weights"]
    best_bias = best_data["bias"]
    best_probs = best_data["val_prob"]

    print(f"\nTop Performing Classifier: {best_model_name} (Validation F1 Score: {best_val_f1:.3f})")
    print(f"Learned Weights: {[round(w, 4) for w in best_weights]}, Bias: {best_bias:.4f}")

    loss_plot_path = str(plots_dir / "classification_loss_curves.png")
    boundary_plot_path = str(plots_dir / "decision_boundary_vibration_temp.png")
    cm_plot_path = str(plots_dir / "confusion_matrix_best_model.png")
    thresh_plot_path = str(plots_dir / "threshold_tradeoffs.png")

    plot_classification_loss_curves(results, loss_plot_path)
    plot_decision_boundary_2d(
        X_raw=val_ds.raw_features,
        y=val_ds.labels,
        scaler_mean=val_ds.scaler_mean,
        scaler_std=val_ds.scaler_std,
        weights=best_weights,
        bias=best_bias,
        feature_idx_x=0,
        feature_idx_y=1,
        feature_names=val_ds.feature_names,
        save_path=boundary_plot_path,
    )
    plot_confusion_matrix(best_data["val_metrics"]["confusion_matrix"], best_model_name, cm_plot_path)

    sweep_res = sweep_thresholds(val_ds.labels, best_probs)
    plot_threshold_tradeoffs(sweep_res, thresh_plot_path)

    print(f"\nSaved plots to:\n > {loss_plot_path}\n > {boundary_plot_path}\n > {cm_plot_path}\n > {thresh_plot_path}")

    report_file = reports_dir / "predict_maint_summary.txt"
    with open(report_file, "w") as f:
        f.write("=" * 75 + "\n")
        f.write("PREDICTMAINT: EXPERIMENTAL SUMMARY REPORT\n")
        f.write("=" * 75 + "\n\n")
        f.write("1. ALGORITHM BENCHMARK\n")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")
        for name, data in results.items():
            vm = data["val_metrics"]
            f.write(f"{name:<20} | {vm['bce']:<10.4f} | {vm['accuracy']:<10.1%} | {vm['precision']:<11.1%} | {vm['recall']:<10.1%} | {vm['f1']:<8.3f} | {data['time']:<10.4f}\n")
        f.write(f"\nBest Model: {best_model_name}\n")
        f.write(f"Confusion Matrix (TN, FP / FN, TP): {best_data['val_metrics']['confusion_matrix']}\n\n")
        f.write("2. THRESHOLD SENSITIVITY SWEEP (Best Model)\n")
        f.write(f"{'Threshold':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<8}\n")
        f.write("-" * 55 + "\n")
        for i, t in enumerate(sweep_res["thresholds"]):
            f.write(f"{t:<10.2f} | {sweep_res['accuracies'][i]:<10.1%} | {sweep_res['precisions'][i]:<10.1%} | {sweep_res['recalls'][i]:<10.1%} | {sweep_res['f1_scores'][i]:<8.3f}\n")

    print(f"Summary report saved to: {report_file}")
    print("=" * 78)


if __name__ == "__main__":
    run_maintenance_experiments()
