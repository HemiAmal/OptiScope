"""
Unit Tests for Convergence Module (core/convergence.py).
"""

import unittest
from shared_optimization.core.convergence import (
    OptimizationResult,
    IterationRecord,
    ConvergenceCriterion,
    check_convergence,
    check_finite,
)


class TestConvergence(unittest.TestCase):
    def test_optimization_result_structure(self):
        rec = IterationRecord(iteration=1, x=[0.1, 0.2], fun=0.05, grad_norm=0.01, step_size=0.1)
        res = OptimizationResult(
            x=[0.0, 0.0],
            fun=0.0,
            grad_norm=1e-6,
            iterations=10,
            func_evals=15,
            grad_evals=10,
            hess_evals=0,
            converged=True,
            status_message="Converged",
            history=[rec],
            execution_time=0.005,
        )
        d = res.to_dict()
        self.assertEqual(d["iterations"], 10)
        self.assertTrue(d["converged"])
        self.assertEqual(len(res.history), 1)

    def test_convergence_criteria_checks(self):
        crit = ConvergenceCriterion(grad_tol=1e-5, step_tol=1e-8, obj_tol=1e-12)

        # 1. Gradient tolerance satisfied
        msg1 = check_convergence(grad_norm=1e-6, criterion=crit)
        self.assertIsNotNone(msg1)
        self.assertIn("gradient norm", msg1)

        # 2. Gradient tolerance NOT satisfied, step tolerance satisfied
        msg2 = check_convergence(grad_norm=1.0, step_norm=1e-9, criterion=crit)
        self.assertIsNotNone(msg2)
        self.assertIn("step size", msg2)

        # 3. Neither satisfied
        msg3 = check_convergence(grad_norm=1.0, step_norm=1.0, criterion=crit)
        self.assertIsNone(msg3)

    def test_check_finite_detection(self):
        check_finite([1.0, 2.0, -3.5])

        with self.assertRaises(FloatingPointError):
            check_finite([1.0, float("nan")])

        with self.assertRaises(FloatingPointError):
            check_finite([float("inf"), 2.0])


if __name__ == "__main__":
    unittest.main()
