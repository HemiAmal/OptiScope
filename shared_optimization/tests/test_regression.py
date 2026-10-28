"""
Unit tests for regression, classification, and adaptive optimizers:
Batch GD, SGD, Mini-batch, Logistic Regression, AdaGrad, RMSProp, and Adam.
"""

import unittest
from shared_optimization.problems.linear_regression import (
    generate_synthetic_linear_data,
    compute_mse_loss,
)
from shared_optimization.problems.logistic_regression import (
    generate_synthetic_classification_data,
    classification_accuracy,
)
from shared_optimization.algorithms.regression_optimizers import (
    train_linear_regression_gd,
    train_linear_regression_sgd,
    train_linear_regression_minibatch,
    train_logistic_regression,
)
from shared_optimization.algorithms.adaptive_optimizers import (
    adagrad_optimizer,
    rmsprop_optimizer,
    adam_optimizer,
)
from shared_optimization.problems.quadratic import f_quadratic, grad_quadratic


class TestRegressionAndAdaptive(unittest.TestCase):
    def setUp(self):
        # Synthetic linear regression dataset: y = 2.5*x0 - 1.5*x1 + 4.0
        self.lin_data = generate_synthetic_linear_data(
            n_samples=100, true_w=[2.5, -1.5], true_b=4.0, noise_amplitude=0.01, seed=42
        )
        # Synthetic classification dataset
        self.clf_data = generate_synthetic_classification_data(n_samples=100, seed=42)

    def test_method_8_linear_regression_gd(self):
        w, b, res = train_linear_regression_gd(self.lin_data, learning_rate=0.08, max_epochs=200)
        self.assertAlmostEqual(w[0], 2.5, places=1)
        self.assertAlmostEqual(w[1], -1.5, places=1)
        self.assertAlmostEqual(b, 4.0, places=1)
        self.assertLess(res.fun, 0.01)

    def test_method_9_linear_regression_sgd(self):
        w, b, res = train_linear_regression_sgd(self.lin_data, learning_rate=0.01, max_epochs=30)
        self.assertLess(res.fun, 0.05)

    def test_method_10_linear_regression_minibatch(self):
        w, b, res = train_linear_regression_minibatch(self.lin_data, batch_size=16, learning_rate=0.03, max_epochs=40)
        self.assertLess(res.fun, 0.02)

    def test_method_11_logistic_regression(self):
        w, b, acc, res = train_logistic_regression(self.clf_data, learning_rate=0.15, max_epochs=250)
        self.assertGreaterEqual(acc, 0.95)
        self.assertLess(res.fun, 0.2)

    def test_method_12_adagrad(self):
        res = adagrad_optimizer(f_quadratic, grad_quadratic, x0=[3.0, 1.0], learning_rate=0.5)
        self.assertLess(res.fun, 0.01)

    def test_method_13_rmsprop(self):
        res = rmsprop_optimizer(f_quadratic, grad_quadratic, x0=[3.0, 1.0], learning_rate=0.01)
        self.assertLess(res.fun, 0.01)

    def test_method_14_adam(self):
        res = adam_optimizer(f_quadratic, grad_quadratic, x0=[3.0, 1.0], learning_rate=0.1)
        self.assertLess(res.fun, 0.01)


if __name__ == "__main__":
    unittest.main()
