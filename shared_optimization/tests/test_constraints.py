"""
Unit tests for nonsmooth and constrained optimization routines:
subgradient methods, Lagrange multipliers, KKT conditions, and LICQ.
"""

import unittest
from shared_optimization.problems.linear_regression import generate_synthetic_linear_data
from shared_optimization.problems.lasso import lasso_objective
from shared_optimization.problems.constrained import (
    create_standard_equality_problem,
    create_standard_kkt_problem,
    ConstrainedProblem,
)
from shared_optimization.algorithms.subgradient import (
    subgradient_method,
    subgradient_lasso,
)
from shared_optimization.algorithms.constrained_optimization import (
    lagrange_multiplier_solver,
    kkt_conditions_solver,
    check_licq,
)


class TestNonsmoothAndConstraints(unittest.TestCase):
    def test_method_15_subgradient_absolute_value(self):
        # f(x) = |x|, minimum is 0.0 at x = 0
        def f_abs(v):
            return abs(v[0])

        def subgrad_abs(v):
            x = v[0]
            if x > 1e-6:
                return [1.0]
            elif x < -1e-6:
                return [-1.0]
            else:
                return [0.0]

        res = subgradient_method(f_abs, subgrad_abs, x0=[5.0], step_size_init=0.5, max_iters=200)
        self.assertLess(res.fun, 0.05)
        self.assertLess(abs(res.x[0]), 0.05)

    def test_method_16_subgradient_lasso(self):
        dataset = generate_synthetic_linear_data(n_samples=50, true_w=[2.0, 0.0], true_b=0.0, seed=42)
        res = subgradient_lasso(dataset, lmbda=0.2, step_size_init=0.2, max_iters=200)
        self.assertLess(res.fun, 1.0)

    def test_method_21_lagrange_multiplier_solver(self):
        # min 0.5 * x^T * Q * x + c^T * x  s.t.  A * x = b
        # Let's map min (x1 - 1)^2 + (x2 - 2)^2 = x1^2 - 2x1 + 1 + x2^2 - 4x2 + 4
        # = 0.5 * (2*x1^2 + 2*x2^2) - 2x1 - 4x2 + 5
        # Q = [[2, 0], [0, 2]], c = [-2, -4]
        # Constraint: x1 + x2 = 1 => A = [[1, 1]], b = [1]
        Q = [[2.0, 0.0], [0.0, 2.0]]
        c = [-2.0, -4.0]
        A = [[1.0, 1.0]]
        b = [1.0]

        res = lagrange_multiplier_solver(Q, c, A, b)
        # Known solution: x* = [0.0, 1.0], lambda* = -2.0
        self.assertAlmostEqual(res["x"][0], 0.0, places=5)
        self.assertAlmostEqual(res["x"][1], 1.0, places=5)
        self.assertAlmostEqual(res["lambda"][0], 2.0, places=5)
        self.assertLess(res["primal_residual"], 1e-6)
        self.assertLess(res["stationarity_residual"], 1e-6)

    def test_method_22_kkt_conditions_solver(self):
        # min x1^2 + x2^2  s.t. 1 - x1 - x2 <= 0
        prob = create_standard_kkt_problem()
        # Optimum is at (0.5, 0.5) with mu = 1.0
        res = kkt_conditions_solver(prob, x=[0.5, 0.5], mu_ineq=[1.0], tol=1e-5)
        self.assertTrue(res["kkt_satisfied"])
        self.assertTrue(res["stationarity_passed"])
        self.assertTrue(res["primal_feasibility_passed"])
        self.assertTrue(res["dual_feasibility_passed"])
        self.assertTrue(res["complementary_slackness_passed"])

        # Non-optimal candidate (e.g. at [2.0, 2.0]) should fail stationarity or complementarity
        res_fail = kkt_conditions_solver(prob, x=[2.0, 2.0], mu_ineq=[1.0], tol=1e-5)
        self.assertFalse(res_fail["kkt_satisfied"])

    def test_method_23_check_licq(self):
        prob = create_standard_kkt_problem()
        # At (0.5, 0.5), g1 = 1 - 0.5 - 0.5 = 0 (Active). Grad is [-1, -1] != 0.
        licq_res = check_licq(prob, x=[0.5, 0.5])
        self.assertTrue(licq_res["licq_satisfied"])
        self.assertEqual(licq_res["num_active_constraints"], 1)

        # Inactive point (2.0, 2.0): g1 = -3.0 < 0 (0 active constraints => LICQ holds trivially)
        licq_inactive = check_licq(prob, x=[2.0, 2.0])
        self.assertTrue(licq_inactive["licq_satisfied"])
        self.assertEqual(licq_inactive["num_active_constraints"], 0)


if __name__ == "__main__":
    unittest.main()
