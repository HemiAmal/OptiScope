"""
Unit tests for ResourceOpt project.
Tests:
- Lagrange multiplier solver for equality constraints
- KKT conditions evaluator
- LICQ rank and Gram determinant checker
- Verification of 5 constrained scenarios
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

from ResourceOpt.problems.production_allocation import ProductionAllocationProblem


class TestResourceOpt(unittest.TestCase):
    def setUp(self):
        self.prob = ProductionAllocationProblem()

    def test_lagrange_multiplier_solver(self):
        sol = self.prob.solve_equality_lagrange(target_demand=8.0)
        x = sol["x"]
        self.assertEqual(len(x), 2)
        self.assertAlmostEqual(x[0] + x[1], 8.0, places=5)
        # Check gradient stationarity with lambda
        lam = sol["lambda"][0] if isinstance(sol["lambda"], list) else sol["lambda"]
        grad = self.prob.grad(x)
        self.assertAlmostEqual(grad[0] + lam, 0.0, places=4)
        self.assertAlmostEqual(grad[1] + lam, 0.0, places=4)

    def test_case1_interior_optimum_kkt(self):
        scenarios = self.prob.test_all_five_scenarios()
        c1 = scenarios["Case 1: Feasible Interior Optimum"]
        kkt = c1["kkt"]
        self.assertTrue(kkt["stationarity_passed"])
        self.assertTrue(kkt["primal_feasibility_passed"])
        self.assertTrue(kkt["dual_feasibility_passed"])
        self.assertTrue(kkt["complementary_slackness_passed"])
        self.assertTrue(kkt["kkt_satisfied"])
        self.assertTrue(c1["licq"]["licq_satisfied"])

    def test_case2_boundary_optimum_kkt(self):
        scenarios = self.prob.test_all_five_scenarios()
        c2 = scenarios["Case 2: Feasible Boundary Optimum"]
        kkt = c2["kkt"]
        self.assertTrue(kkt["stationarity_passed"])
        self.assertTrue(kkt["primal_feasibility_passed"])
        self.assertTrue(kkt["dual_feasibility_passed"])
        self.assertTrue(kkt["complementary_slackness_passed"])
        self.assertTrue(kkt["kkt_satisfied"])
        self.assertTrue(c2["licq"]["licq_satisfied"])

    def test_case3_infeasible_candidate(self):
        scenarios = self.prob.test_all_five_scenarios()
        c3 = scenarios["Case 3: Infeasible Candidate Point"]
        kkt = c3["kkt"]
        # Storage capacity x1 <= 6.0 violated by 6.5
        self.assertFalse(kkt["primal_feasibility_passed"])
        self.assertFalse(kkt["kkt_satisfied"])

    def test_case4_and_5_licq_qualification(self):
        scenarios = self.prob.test_all_five_scenarios()
        c4 = scenarios["Case 4: Linearly Independent Active Constraints"]
        c5 = scenarios["Case 5: Linearly Dependent Active Constraints"]
        
        # Case 4 should pass LICQ
        self.assertTrue(c4["licq_satisfied"])
        self.assertGreater(c4["gram_det"], 0.01)
        
        # Case 5 should fail LICQ due to rank deficiency
        self.assertFalse(c5["licq_satisfied"])
        self.assertAlmostEqual(c5["gram_det"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
