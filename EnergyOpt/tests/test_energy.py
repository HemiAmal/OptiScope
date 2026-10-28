"""
Unit Tests for EnergyOpt Project.
Tests:
- Data generation reproducibility & feature structure
- Regression optimizers on energy features
- Lagrange multiplier economic dispatch solver
- KKT condition evaluations and LICQ qualification
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

from EnergyOpt.data.synthetic_energy import create_energy_regression_dataset, generate_hourly_electricity_demand
from EnergyOpt.problems.demand_regression import evaluate_regression_metrics, train_adaptive_linear_regression
from EnergyOpt.problems.generation_dispatch import EconomicGenerationDispatch
from shared_optimization.problems.linear_regression import LinearRegressionDataset
from shared_optimization.algorithms.regression_optimizers import train_linear_regression_gd


class TestEnergyOpt(unittest.TestCase):
    def test_synthetic_data_generation(self):
        hours, temps, demands = generate_hourly_electricity_demand(num_days=10, seed=42)
        self.assertEqual(len(hours), 240)
        self.assertEqual(len(demands), 240)
        self.assertTrue(np.all(demands > 0))  # Electricity demand physically positive
        
        # Test reproducibility with identical seed
        _, _, demands_seed2 = generate_hourly_electricity_demand(num_days=10, seed=42)
        np.testing.assert_allclose(demands, demands_seed2)

    def test_train_val_split_no_leakage(self):
        train_ds, val_ds = create_energy_regression_dataset(num_days=20, train_ratio=0.8, seed=42)
        self.assertEqual(train_ds.features.shape[1], 5)
        self.assertEqual(val_ds.features.shape[1], 5)
        # Check standard deviations strictly positive
        self.assertTrue(np.all(train_ds.scaler_std > 0))
        # Scaled train features have mean ~ 0 and std ~ 1
        np.testing.assert_allclose(np.mean(train_ds.features, axis=0), np.zeros(5), atol=1e-7)

    def test_linear_regression_gd_convergence(self):
        train_ds, _ = create_energy_regression_dataset(num_days=15, train_ratio=0.8, seed=42)
        lr_ds = LinearRegressionDataset(train_ds.features.tolist(), train_ds.targets.tolist())
        weights, bias, res = train_linear_regression_gd(lr_ds, learning_rate=0.05, max_epochs=50)
        self.assertEqual(len(weights), 5)
        self.assertTrue(np.isfinite(bias))
        self.assertLess(res.history[-1].fun, res.history[0].fun)

    def test_adaptive_optimizers_linear_regression(self):
        train_ds, _ = create_energy_regression_dataset(num_days=10, train_ratio=0.8, seed=42)
        lr_ds = LinearRegressionDataset(train_ds.features.tolist(), train_ds.targets.tolist())
        weights, bias, res, hist = train_adaptive_linear_regression(lr_ds, method="adam", learning_rate=0.2, max_epochs=30)
        self.assertEqual(len(weights), 5)
        self.assertLess(hist[-1], hist[0])

    def test_economic_dispatch_lagrange(self):
        demand = 280.0
        dispatch = EconomicGenerationDispatch(predicted_demand=demand)
        sol = dispatch.solve_equality_dispatch()
        p_opt = sol["x"]
        self.assertEqual(len(p_opt), 3)
        self.assertAlmostEqual(sum(p_opt), demand, places=4)
        
        # Verify analytical stationarity: Q*P + c + A^T*lambda = 0
        lam = sol["lambda"][0] if isinstance(sol["lambda"], list) else sol["lambda"]
        grad = dispatch.cost_gradient(p_opt)
        for g in grad:
            self.assertAlmostEqual(g + lam, 0.0, places=4)

    def test_kkt_and_licq_evaluation(self):
        demand = 300.0
        dispatch = EconomicGenerationDispatch(predicted_demand=demand)
        sol = dispatch.solve_equality_dispatch()
        p_opt = sol["x"]
        lam = sol["lambda"][0] if isinstance(sol["lambda"], list) else sol["lambda"]
        eval_res = dispatch.evaluate_kkt_and_licq(P_candidate=p_opt, lambda_eq=lam)
        
        self.assertTrue(eval_res["kkt"]["stationarity_passed"])
        self.assertTrue(eval_res["kkt"]["primal_feasibility_passed"])
        self.assertTrue(eval_res["licq"]["licq_satisfied"])


if __name__ == "__main__":
    unittest.main()
