"""
Production and resource allocation under constraints.

Mathematical Formulation:
  min f(x) = 1/2 x^T Q x + c^T x
  subject to:
    h(x) = A_eq x - b_eq = 0        (Production target commitment)
    g(x) = G x - h_ineq <= 0        (Machine hours, storage capacity, non-negativity)
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared_optimization.problems.constrained import ConstrainedProblem
from shared_optimization.algorithms.constrained_optimization import (
    lagrange_multiplier_solver,
    kkt_conditions_solver,
    check_licq,
)


class ProductionAllocationProblem:
    """
    Continuous 2-product manufacturing allocation model.
    Cost: f(x1, x2) = 1/2 (4 x1^2 + 6 x2^2) - 20 x1 - 30 x2
    """
    def __init__(self):
        self.Q = np.array([[4.0, 0.0], [0.0, 6.0]], dtype=float)
        self.c = np.array([-20.0, -30.0], dtype=float)

    def f(self, x: List[float]) -> float:
        xa = np.array(x, dtype=float)
        return float(0.5 * xa.T @ self.Q @ xa + self.c.T @ xa)

    def grad(self, x: List[float]) -> List[float]:
        xa = np.array(x, dtype=float)
        return (self.Q @ xa + self.c).tolist()

    def solve_equality_lagrange(self, target_demand: float = 8.0) -> Dict[str, Any]:
        """
        Solves the equality constrained system x1 + x2 = target_demand via Lagrange multipliers.
        """
        A = [[1.0, 1.0]]
        b = [float(target_demand)]
        sol = lagrange_multiplier_solver(self.Q.tolist(), self.c.tolist(), A, b)
        return sol

    def create_production_problem(
        self,
        target_demand: float = 8.0,
        machine_limit: float = 10.0,
        storage_limit_x1: float = 6.0,
        include_redundant_equality: bool = False,
    ) -> ConstrainedProblem:
        """
        Creates ConstrainedProblem instance with standard manufacturing constraints.
        """
        # Equality: x1 + x2 - target = 0
        eq_cons = [lambda x, d=target_demand: (x[0] + x[1]) - d]
        eq_grads = [lambda x: [1.0, 1.0]]

        if include_redundant_equality:
            # Add duplicate constraint 2 x1 + 2 x2 - 2 target = 0 (causes LICQ failure)
            eq_cons.append(lambda x, d=target_demand: 2.0 * (x[0] + x[1]) - 2.0 * d)
            eq_grads.append(lambda x: [2.0, 2.0])

        # Inequalities:
        # 1. Machine capacity: 1.5 x1 + x2 - machine_limit <= 0
        # 2. Storage capacity: x1 - storage_limit <= 0
        # 3. Non-negativity x1 >= 0 -> -x1 <= 0
        # 4. Non-negativity x2 >= 0 -> -x2 <= 0
        ineq_cons = [
            lambda x, ml=machine_limit: 1.5 * x[0] + x[1] - ml,
            lambda x, sl=storage_limit_x1: x[0] - sl,
            lambda x: -x[0],
            lambda x: -x[1],
        ]
        ineq_grads = [
            lambda x: [1.5, 1.0],
            lambda x: [1.0, 0.0],
            lambda x: [-1.0, 0.0],
            lambda x: [0.0, -1.0],
        ]

        return ConstrainedProblem(
            f=self.f,
            grad_f=self.grad,
            eq_constraints=eq_cons,
            eq_gradients=eq_grads,
            ineq_constraints=ineq_cons,
            ineq_gradients=ineq_grads,
            name="Production & Resource Allocation Problem",
        )

    def test_all_five_scenarios(self) -> Dict[str, Any]:
        """
        Systematically runs all 5 required test cases:
        1. Valid Feasible Interior Optimum
        2. Valid Feasible Boundary Optimum
        3. Infeasible Candidate Point
        4. Linearly Independent Active Constraints (LICQ Success)
        5. Linearly Dependent / Redundant Active Constraints (LICQ Failure)
        """
        results = {}

        # Case 1: Feasible Interior Optimum (Demand = 8.0, capacity inactive)
        p1 = self.create_production_problem(target_demand=8.0, machine_limit=10.0)
        sol1 = self.solve_equality_lagrange(target_demand=8.0)
        x1 = sol1["x"]
        lam1 = sol1["lambda"][0] if isinstance(sol1["lambda"], list) else sol1["lambda"]
        kkt1 = kkt_conditions_solver(p1, x1, lambda_eq=[lam1], mu_ineq=[0.0, 0.0, 0.0, 0.0])
        licq1 = check_licq(p1, x1)
        results["Case 1: Feasible Interior Optimum"] = {
            "x": x1,
            "lambda": [lam1],
            "mu": [0.0, 0.0, 0.0, 0.0],
            "f_val": self.f(x1),
            "kkt": kkt1,
            "licq": licq1,
            "verdict": "Verified Optimum" if kkt1["kkt_satisfied"] and licq1["licq_satisfied"] else "Not Optimum",
        }

        # Case 2: Feasible Boundary Optimum (Machine limit binds at x* = [2.0, 7.0])
        p2 = self.create_production_problem(target_demand=9.0, machine_limit=10.0)
        x2 = [2.0, 7.0]
        lam2 = -60.0
        mu2 = [48.0, 0.0, 0.0, 0.0]
        kkt2 = kkt_conditions_solver(p2, x2, lambda_eq=[lam2], mu_ineq=mu2)
        licq2 = check_licq(p2, x2)
        results["Case 2: Feasible Boundary Optimum"] = {
            "x": x2,
            "lambda": [lam2],
            "mu": mu2,
            "f_val": self.f(x2),
            "kkt": kkt2,
            "licq": licq2,
            "verdict": "Verified Optimum" if kkt2["kkt_satisfied"] and licq2["licq_satisfied"] else "Not Optimum",
        }

        # Case 3: Infeasible Candidate Point (Storage limit violated)
        p3 = self.create_production_problem(target_demand=10.0, storage_limit_x1=6.0)
        x3 = [6.5, 3.5]
        kkt3 = kkt_conditions_solver(p3, x3, lambda_eq=[0.0], mu_ineq=[0.0, 0.0, 0.0, 0.0])
        licq3 = check_licq(p3, x3)
        results["Case 3: Infeasible Candidate Point"] = {
            "x": x3,
            "lambda": [0.0],
            "mu": [0.0, 0.0, 0.0, 0.0],
            "f_val": self.f(x3),
            "kkt": kkt3,
            "licq": licq3,
            "verdict": "Infeasible Candidate" if not kkt3["primal_feasibility_passed"] else "Feasible",
        }

        # Case 4: Linearly Independent Active Constraints (LICQ holds)
        results["Case 4: Linearly Independent Active Constraints"] = {
            "x": x2,
            "num_active": licq2["num_active_constraints"],
            "gram_det": licq2["gram_determinant"],
            "licq_satisfied": licq2["licq_satisfied"],
            "verdict": "LICQ Holds (Full Rank Active Jacobian)",
        }

        # Case 5: Linearly Dependent / Redundant Constraints (LICQ fails)
        p5 = self.create_production_problem(target_demand=9.0, include_redundant_equality=True)
        licq5 = check_licq(p5, [2.0, 7.0])
        results["Case 5: Linearly Dependent Active Constraints"] = {
            "x": [2.0, 7.0],
            "num_active": licq5["num_active_constraints"],
            "gram_det": licq5["gram_determinant"],
            "licq_satisfied": licq5["licq_satisfied"],
            "verdict": "LICQ Failed (Rank Deficient Redundant Constraints)" if not licq5["licq_satisfied"] else "LICQ Holds",
        }

        return results
