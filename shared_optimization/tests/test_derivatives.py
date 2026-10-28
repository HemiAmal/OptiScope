"""
Unit Tests for Derivatives Module (core/derivatives.py).
"""

import unittest
from shared_optimization.core.derivatives import (
    finite_difference_gradient,
    finite_difference_hessian,
    finite_difference_hessian_from_grad,
    verify_gradient,
    verify_hessian,
)


class TestDerivatives(unittest.TestCase):
    def setUp(self):
        # Test function: f(x, y) = x^2 + 3*x*y + 4*y^2
        self.f = lambda v: v[0]**2 + 3.0 * v[0] * v[1] + 4.0 * v[1]**2
        self.grad_f = lambda v: [2.0 * v[0] + 3.0 * v[1], 3.0 * v[0] + 8.0 * v[1]]
        self.hess_f = lambda v: [[2.0, 3.0], [3.0, 8.0]]

    def test_gradient_verification_polynomial(self):
        pts = [[0.0, 0.0], [1.5, -2.0], [10.0, 5.0]]
        for pt in pts:
            res = verify_gradient(self.f, self.grad_f, pt, h=1e-6)
            self.assertTrue(res["passed"])
            self.assertLess(res["max_abs_error"], 1e-4)

    def test_hessian_verification_polynomial(self):
        pts = [[0.0, 0.0], [2.0, -1.0], [5.0, 5.0]]
        for pt in pts:
            res = verify_hessian(self.f, self.hess_f, pt, grad_f=self.grad_f, h=1e-4)
            self.assertTrue(res["passed"])
            self.assertLess(res["max_abs_error"], 1e-3)

    def test_invalid_step_sizes(self):
        with self.assertRaises(ValueError):
            finite_difference_gradient(self.f, [1.0, 1.0], h=0.0)
        with self.assertRaises(ValueError):
            finite_difference_hessian(self.f, [1.0, 1.0], h=-1e-4)


if __name__ == "__main__":
    unittest.main()
