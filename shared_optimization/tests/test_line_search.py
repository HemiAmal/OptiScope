"""
Unit tests for line search module: exact quadratic line search,
Armijo condition, and geometric backtracking.
"""

import unittest
from shared_optimization.algorithms.line_search import (
    exact_line_search_quadratic,
    backtracking_line_search,
    armijo_condition,
)
from shared_optimization.core.vectors import norm_l2


class TestLineSearch(unittest.TestCase):
    def test_method_2_exact_line_search_quadratic(self):
        # f(x, y) = 0.5 * (x^2 + 100*y^2), Q = [[1, 0], [0, 100]]
        Q = [[1.0, 0.0], [0.0, 100.0]]
        x = [2.0, 1.0]
        # grad = [2.0, 100.0]
        grad = [2.0, 100.0]
        # p = -grad = [-2.0, -100.0]
        p = [-2.0, -100.0]

        # Exact alpha* = <g, g> / <g, Q*g>
        # <g, g> = 4 + 10000 = 10004
        # Q*g = [2, 10000]
        # <g, Q*g> = 2*2 + 100*10000 = 4 + 1000000 = 1000004
        # alpha* = 10004 / 1000004 approx 0.01000396
        expected_alpha = 10004.0 / 1000004.0
        alpha = exact_line_search_quadratic(x, p, Q, grad=grad)
        self.assertAlmostEqual(alpha, expected_alpha, places=7)

    def test_method_4_armijo_condition(self):
        # f(x) = x^2, df/dx = 2x. At x = 2: f(2) = 4, g = 4. Search dir p = -4
        f = lambda v: v[0]**2
        x = [2.0]
        p = [-4.0]
        f_x = 4.0
        grad_x = [4.0]

        # Directional derivative = <g, p> = -16.
        # Armijo threshold = 4 + c1 * alpha * (-16)
        # For small alpha = 0.1: x_trial = 2 - 0.4 = 1.6, f_trial = 2.56.
        # Threshold with c1=1e-4: 4 - 0.0016 = 3.9984. 2.56 <= 3.9984 -> True!
        self.assertTrue(armijo_condition(f, x, p, alpha=0.1, f_x=f_x, grad_x=grad_x, c1=1e-4))

        # For giant overshoot alpha = 2.0: x_trial = 2 - 8 = -6, f_trial = 36.
        # Threshold: 4 - 1e-4 * 2 * 16 = 3.9968. 36 <= 3.9968 -> False!
        self.assertFalse(armijo_condition(f, x, p, alpha=2.0, f_x=f_x, grad_x=grad_x, c1=1e-4))

    def test_method_3_backtracking_line_search(self):
        f = lambda v: v[0]**2
        x = [2.0]
        p = [-4.0]
        f_x = 4.0
        grad_x = [4.0]

        def armijo_check(trial_alpha: float) -> bool:
            return armijo_condition(f, x, p, trial_alpha, f_x, grad_x, c1=1e-4)

        # Starts with overshoot alpha_init=2.0, should backtrack until alpha <= 0.5
        alpha, evals = backtracking_line_search(f, x, p, acceptance_fn=armijo_check, alpha_init=2.0, rho=0.5)
        self.assertTrue(evals >= 1)
        self.assertLessEqual(alpha, 0.5)


if __name__ == "__main__":
    unittest.main()
