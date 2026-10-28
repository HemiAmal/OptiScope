"""
Unit Tests for PredictMaint Project.
Tests:
- Sensor telemetry synthesis and label statistics
- Zero-leakage standard scaling
- Metric calculations (BCE, Accuracy, Precision, Recall, F1)
- Logistic regression optimizers convergence
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
LAB_ROOT = PROJECT_DIR.parent
for p in [str(PROJECT_DIR), str(LAB_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from PredictMaint.data.synthetic_sensors import generate_industrial_sensor_data, create_predictive_maintenance_dataset
from PredictMaint.problems.failure_classification import (
    compute_classification_metrics,
    sweep_thresholds,
    train_adaptive_logistic_regression,
)
from shared_optimization.problems.logistic_regression import LogisticRegressionDataset
from shared_optimization.algorithms.regression_optimizers import train_logistic_regression


class TestPredictMaint(unittest.TestCase):
    def test_synthetic_sensor_generation(self):
        X, y, names, w_true = generate_industrial_sensor_data(num_samples=200, seed=42)
        self.assertEqual(X.shape, (200, 5))
        self.assertEqual(len(y), 200)
        self.assertTrue(set(np.unique(y)).issubset({0.0, 1.0}))
        
        # Test reproducibility
        X2, y2, _, _ = generate_industrial_sensor_data(num_samples=200, seed=42)
        np.testing.assert_allclose(X, X2)
        np.testing.assert_allclose(y, y2)

    def test_train_val_split_zero_leakage(self):
        train_ds, val_ds = create_predictive_maintenance_dataset(num_samples=400, train_ratio=0.75, seed=42)
        self.assertEqual(train_ds.features.shape, (300, 5))
        self.assertEqual(val_ds.features.shape, (100, 5))
        np.testing.assert_allclose(np.mean(train_ds.features, axis=0), np.zeros(5), atol=1e-7)

    def test_classification_metrics(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.4, 0.2, 0.8])  # At 0.5: pred = [1, 0, 0, 1] -> TP=1, FN=1, TN=1, FP=1
        m = compute_classification_metrics(y_true, y_prob, threshold=0.5)
        self.assertEqual(m["accuracy"], 0.5)
        self.assertEqual(m["precision"], 0.5)
        self.assertEqual(m["recall"], 0.5)
        self.assertEqual(m["f1"], 0.5)
        self.assertEqual(m["confusion_matrix"], [[1, 1], [1, 1]])

    def test_logistic_regression_gd(self):
        train_ds, _ = create_predictive_maintenance_dataset(num_samples=250, train_ratio=0.8, seed=42)
        log_ds = LogisticRegressionDataset(train_ds.features.tolist(), train_ds.labels.tolist())
        weights, bias, acc, res = train_logistic_regression(log_ds, learning_rate=0.2, max_epochs=60)
        self.assertEqual(len(weights), 5)
        self.assertTrue(np.isfinite(bias))
        self.assertLess(res.history[-1].fun, res.history[0].fun)

    def test_adaptive_logistic_regression(self):
        train_ds, _ = create_predictive_maintenance_dataset(num_samples=250, train_ratio=0.8, seed=42)
        weights, bias, res, hist = train_adaptive_logistic_regression(
            train_ds.features, train_ds.labels, method="adam", learning_rate=0.1, max_epochs=40
        )
        self.assertEqual(len(weights), 5)
        self.assertLess(hist[-1], hist[0])


if __name__ == "__main__":
    unittest.main()
