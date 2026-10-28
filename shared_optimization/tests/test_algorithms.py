"""
Unit tests for smooth optimization algorithms: fixed GD, exact quadratic GD,
Armijo GD, Polyak momentum, Nesterov, Newton, damped Newton, and BFGS.
"""

import unittest
from shared_optimization.core.vectors import norm_l2
from shared_optimization.core.convergence import ConvergenceCriterion
from shared_optimization.problems.quadratic import f_quadratic, grad_quadratic, hess_quadratic
from shared_optimization.problems.rosenbrock import f_rosenbrock, grad_rosenbrock, hess_rosenbrock
from shared_optimization.algorithms.gradient_descent import (
    gradient_descent_fixed,
    gradient_descent_exact_quadratic,
    gradient_descent_armijo,
)
from shared_optimization.algorithms.momentum import momentum_gd
from shared_optimization.algorithms.nesterov import nesterov_gd
from shared_optimization.algorithms.newton import (
    newton_method,
    damped_newton_method,
    verify_newton_one_step,
)
from shared_optimization.algorithms.bfgs import bfgs_optimizer


class TestSmoothAlgorithms(unittest.TestCase):
    def test_method_1_gradient_descent_fixed(self):
        # Quadratic f(x, y) = 0.5 * (x^2 + 100*y^2)
        # Max stable learning rate for lambda_max = 100 is alpha < 2 / 100 = 0.02
        res = gradient_descent_fixed(
            f=f_quadratic,
            grad_f=grad_quadratic,
            x0=[2.0, 1.0],
            learning_rate=0.015,
            criterion=ConvergenceCriterion(max_iters=1000, grad_tol=1e-4),
        )
        self.assertLess(res.grad_norm, 1e-4)
        self.assertLess(res.fun, 1e-5)

    def test_method_2_gradient_descent_exact_quadratic(self):
        Q = [[1.0, 0.0], [0.0, 100.0]]
        res = gradient_descent_exact_quadratic(
            x0=[2.0, 1.0],
            Q=Q,
            criterion=ConvergenceCriterion(max_iters=500, grad_tol=1e-4),
        )
        self.assertTrue(res.converged)
        self.assertLess(res.fun, 1e-6)

    def test_method_5_gradient_descent_armijo(self):
        res = gradient_descent_armijo(
            f=f_quadratic,
            grad_f=grad_quadratic,
            x0=[2.0, 1.0],
            alpha_init=1.0,
            rho=0.5,
            criterion=ConvergenceCriterion(max_iters=1000, grad_tol=1e-4),
        )
        self.assertLess(res.grad_norm, 1e-4)
        self.assertLess(res.fun, 1e-5)

    def test_method_6_momentum_gd(self):
        res = momentum_gd(
            f=f_quadratic,
            grad_f=grad_quadratic,
            x0=[2.0, 1.0],
            learning_rate=0.005,
            beta=0.8,
            criterion=ConvergenceCriterion(max_iters=1000, grad_tol=1e-3),
        )
        self.assertLess(res.fun, 1e-4)

    def test_method_7_nesterov_gd(self):
        res = nesterov_gd(
            f=f_quadratic,
            grad_f=grad_quadratic,
            x0=[2.0, 1.0],
            learning_rate=0.005,
            beta=0.8,
            criterion=ConvergenceCriterion(max_iters=1000, grad_tol=1e-3),
        )
        self.assertLess(res.fun, 1e-4)

    def test_method_17_newton_quadratic(self):
        # Pure Newton on quadratic should reach optimum in exactly 1 iteration!
        res = newton_method(
            f=f_quadratic,
            grad_f=grad_quadratic,
            hess_f=hess_quadratic,
            x0=[10.0, -5.0],
            criterion=ConvergenceCriterion(max_iters=5, grad_tol=1e-9),
        )
        self.assertTrue(res.converged)
        self.assertEqual(res.iterations, 1)
        self.assertAlmostEqual(res.x[0], 0.0, places=9)
        self.assertAlmostEqual(res.x[1], 0.0, places=9)
        self.assertAlmostEqual(res.fun, 0.0, places=9)

    def test_method_18_damped_newton_rosenbrock(self):
        # Rosenbrock is non-convex far from optimum; damped Newton handles the curvature
        res = damped_newton_method(
            f=f_rosenbrock,
            grad_f=grad_rosenbrock,
            hess_f=hess_rosenbrock,
            x0=[-1.2, 1.0],
            criterion=ConvergenceCriterion(max_iters=100, grad_tol=1e-5),
        )
        self.assertTrue(res.converged)
        self.assertAlmostEqual(res.x[0], 1.0, places=4)
        self.assertAlmostEqual(res.x[1], 1.0, places=4)
        self.assertAlmostEqual(res.fun, 0.0, places=6)

    def test_method_19_verify_newton_one_step(self):
        Q = [[4.0, 1.0], [1.0, 3.0]]
        c = [-2.0, 5.0]
        x0 = [100.0, -50.0]
        res = verify_newton_one_step(Q=Q, c=c, x0=x0, tol=1e-10)
        self.assertTrue(res["passed"])
        self.assertLess(res["error_norm"], 1e-10)

    def test_method_20_bfgs_quadratic_and_rosenbrock(self):
        # On Quadratic
        res_quad = bfgs_optimizer(
            f=f_quadratic,
            grad_f=grad_quadratic,
            x0=[5.0, 2.0],
            criterion=ConvergenceCriterion(max_iters=100, grad_tol=1e-5),
        )
        self.assertTrue(res_quad.converged)
        self.assertAlmostEqual(res_quad.fun, 0.0, places=6)

        # On Rosenbrock
        res_rosen = bfgs_optimizer(
            f=f_rosenbrock,
            grad_f=grad_rosenbrock,
            x0=[-1.2, 1.0],
            criterion=ConvergenceCriterion(max_iters=200, grad_tol=1e-4),
        )
        self.assertTrue(res_rosen.converged)
        self.assertAlmostEqual(res_rosen.x[0], 1.0, places=3)
        self.assertAlmostEqual(res_rosen.x[1], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
