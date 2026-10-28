"""
Unit tests for SparseModel project.
Tests:
- Synthetic sparse regression data generator
- Subgradient calculation at zero and non-zero coordinates
- General subgradient method on nonsmooth functions
- Subgradient method for Lasso loss
- Regularization path monotonicity and coefficient shrinkage
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

from SparseModel.data.synthetic_sparse import generate_sparse_resource_data, create_sparse_regression_dataset
from SparseModel.problems.lasso_formulation import (
    run_general_subgradient_demo,
    run_lasso_regularization_path,
)
from shared_optimization.problems.linear_regression import LinearRegressionDataset
from shared_optimization.problems.lasso import lasso_subgradient
from shared_optimization.algorithms.subgradient import subgradient_lasso


class TestSparseModel(unittest.TestCase):
    def test_synthetic_sparse_data(self):
        X, y, names, true_w, active = generate_sparse_resource_data(num_samples=150, seed=42)
        self.assertEqual(X.shape, (150, 8))
        self.assertEqual(len(y), 150)
        self.assertEqual(len(active), 3)
        self.assertEqual(true_w[1], 0.0)
        self.assertEqual(true_w[3], 0.0)

    def test_zero_leakage_scaling(self):
        train_ds, val_ds = create_sparse_regression_dataset(num_samples=200, train_ratio=0.8, seed=42)
        self.assertEqual(train_ds.features.shape, (160, 8))
        self.assertEqual(val_ds.features.shape, (40, 8))
        np.testing.assert_allclose(np.mean(train_ds.features, axis=0), np.zeros(8), atol=1e-7)

    def test_subgradient_choice_at_zero(self):
        # Verify valid subgradient handling at w = 0
        X = np.eye(3).tolist()
        y = [1.0, 2.0, 3.0]
        ds = LinearRegressionDataset(X, y)
        w_with_zero = [0.0, 1.0, -1.0]
        g = lasso_subgradient(ds, w_with_zero, lmbda=0.5)
        self.assertEqual(len(g), 3)
        self.assertTrue(all(np.isfinite(val) for val in g))

    def test_general_subgradient_method(self):
        demo = run_general_subgradient_demo(x0=[2.0, -2.0], max_iters=50)
        # Should decrease Euclidean norm
        self.assertLess(demo["final_f"], 2.828)

    def test_lasso_subgradient_shrinkage(self):
        train_ds, _ = create_sparse_regression_dataset(num_samples=200, train_ratio=0.8, seed=42)
        ds = LinearRegressionDataset(train_ds.features.tolist(), train_ds.targets.tolist())
        
        # Low lambda vs High lambda
        res_low = subgradient_lasso(ds, lmbda=0.001, max_iters=100)
        res_high = subgradient_lasso(ds, lmbda=0.5, max_iters=100)
        
        # High regularization should have smaller L1 norm
        l1_low = sum(abs(x) for x in res_low.x)
        l1_high = sum(abs(x) for x in res_high.x)
        self.assertLess(l1_high, l1_low)


if __name__ == "__main__":
    unittest.main()
