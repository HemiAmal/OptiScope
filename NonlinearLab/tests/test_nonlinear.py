"""
Unit Tests for NonlinearLab Project.
Tests:
- Ill-conditioned quadratic objective, analytical gradients and Hessians
- Rosenbrock function, gradient and Hessian
- Convergence of Fixed GD, Exact GD, Armijo GD, Momentum, Nesterov
- Pure Newton and Damped Newton convergence
- Newton One-Step Theorem verification
- BFGS Quasi-Newton convergence
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

from NonlinearLab.problems.quadratic_benchmark import IllConditionedQuadratic
from NonlinearLab.problems.rosenbrock_calibration import RosenbrockCalibration
from shared_optimization.core.derivatives import verify_gradient, verify_hessian


class TestNonlinearLab(unittest.TestCase):
    def test_quadratic_derivatives(self):
        quad = IllConditionedQuadratic(cond_ratio=10.0)
        x_test = [1.5, -2.0]
        grad_verified = verify_gradient(quad.f, quad.grad, x_test, rtol=1e-4, atol=1e-4)
        self.assertTrue(grad_verified["passed"])
        hess_verified = verify_hessian(quad.f, quad.hess, x_test, grad_f=quad.grad, rtol=1e-3, atol=1e-3)
        self.assertTrue(hess_verified["passed"])

    def test_rosenbrock_derivatives(self):
        rosen = RosenbrockCalibration()
        x_test = [-0.5, 1.2]
        grad_verified = verify_gradient(rosen.f, rosen.grad, x_test, rtol=1e-3, atol=1e-3)
        self.assertTrue(grad_verified["passed"])
        hess_verified = verify_hessian(rosen.f, rosen.hess, x_test, grad_f=rosen.grad, rtol=1e-2, atol=1e-2)
        self.assertTrue(hess_verified["passed"])

    def test_newton_one_step_convergence(self):
        quad = IllConditionedQuadratic(cond_ratio=15.0)
        x0 = [7.2, -4.8]
        res = quad.run_newton_one_step_demonstration(x0=x0)
        self.assertTrue(res["passed"])
        self.assertLess(res["error_norm"], 1e-10)

    def test_quadratic_solvers_progress(self):
        quad = IllConditionedQuadratic(cond_ratio=5.0)
        solvers = quad.run_all_quadratic_solvers(x0=[-2.0, 2.0], max_iters=50)
        for name, data in solvers.items():
            res = data["result"]
            # All algorithms should decrease function value
            self.assertLess(res.fun, quad.f([-2.0, 2.0]))

    def test_bfgs_rosenbrock_progress(self):
        rosen = RosenbrockCalibration()
        solvers = rosen.run_all_rosenbrock_solvers(x0=[-1.2, 1.0], max_iters=40)
        res_bfgs = solvers["BFGS"]["result"]
        self.assertLess(res_bfgs.fun, rosen.f([-1.2, 1.0]))


if __name__ == "__main__":
    unittest.main()
