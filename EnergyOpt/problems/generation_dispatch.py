import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared_optimization.problems.constrained import ConstrainedProblem
from shared_optimization.algorithms.constrained_optimization import (
    lagrange_multiplier_solver,
    kkt_conditions_solver,
    check_licq,
)


class EconomicGenerationDispatch:
    def __init__(self, predicted_demand: float = 300.0):
        self.predicted_demand = float(predicted_demand)
        self.a = np.array([0.08, 0.10, 0.12])
        self.b = np.array([12.0, 14.0, 16.0])
        self.Q = np.diag(self.a)
        self.c = self.b
        self.p_min = np.array([50.0, 20.0, 10.0])
        self.p_max = np.array([200.0, 150.0, 100.0])

    def cost_function(self, P: List[float]) -> float:
        p_arr = np.array(P)
        return float(0.5 * p_arr.T @ self.Q @ p_arr + self.c.T @ p_arr)

    def cost_gradient(self, P: List[float]) -> List[float]:
        p_arr = np.array(P)
        return (self.Q @ p_arr + self.c).tolist()

    def solve_equality_dispatch(self, demand: float = None) -> Dict[str, Any]:
        d = self.predicted_demand if demand is None else float(demand)
        A = [[1.0, 1.0, 1.0]]
        b_vec = [d]
        return lagrange_multiplier_solver(
            Q=self.Q.tolist(),
            c=self.c.tolist(),
            A=A,
            b=b_vec,
        )

    def create_constrained_problem(self, demand: float = None) -> ConstrainedProblem:
        d = self.predicted_demand if demand is None else float(demand)

        eq_cons = [lambda P, d=d: sum(P) - d]
        eq_grads = [lambda P: [1.0, 1.0, 1.0]]

        ineq_cons = [
            lambda P: P[0] - self.p_max[0],
            lambda P: P[1] - self.p_max[1],
            lambda P: P[2] - self.p_max[2],
            lambda P: self.p_min[0] - P[0],
            lambda P: self.p_min[1] - P[1],
            lambda P: self.p_min[2] - P[2],
        ]
        ineq_grads = [
            lambda P: [1.0, 0.0, 0.0],
            lambda P: [0.0, 1.0, 0.0],
            lambda P: [0.0, 0.0, 1.0],
            lambda P: [-1.0, 0.0, 0.0],
            lambda P: [0.0, -1.0, 0.0],
            lambda P: [0.0, 0.0, -1.0],
        ]

        return ConstrainedProblem(
            f=self.cost_function,
            grad_f=self.cost_gradient,
            eq_constraints=eq_cons,
            eq_gradients=eq_grads,
            ineq_constraints=ineq_cons,
            ineq_gradients=ineq_grads,
            name=f"Economic Dispatch (Demand={d:.1f} MW)",
        )

    def evaluate_kkt_and_licq(
        self,
        P_candidate: List[float],
        lambda_eq: float,
        mu_ineq: List[float] = None,
        demand: float = None,
    ) -> Dict[str, Any]:
        d = self.predicted_demand if demand is None else float(demand)
        problem = self.create_constrained_problem(demand=d)
        if mu_ineq is None:
            mu_ineq = [0.0] * 6

        kkt_result = kkt_conditions_solver(
            problem=problem,
            x=P_candidate,
            lambda_eq=[lambda_eq],
            mu_ineq=mu_ineq,
            tol=1e-3,
        )
        licq_result = check_licq(
            problem=problem,
            x=P_candidate,
            active_tol=1e-3,
        )

        return {
            "P": P_candidate,
            "cost": self.cost_function(P_candidate),
            "demand": d,
            "kkt": kkt_result,
            "licq": licq_result,
        }

    def analyze_prediction_error_impact(
        self,
        true_demand: float,
        predicted_demand: float,
    ) -> Dict[str, Any]:
        plan_pred = self.solve_equality_dispatch(demand=predicted_demand)
        P_planned = plan_pred["x"]

        plan_true = self.solve_equality_dispatch(demand=true_demand)
        P_ideal = plan_true["x"]

        planned_total_gen = sum(P_planned)
        shortfall = true_demand - planned_total_gen

        planned_cost = self.cost_function(P_planned)
        ideal_cost = self.cost_function(P_ideal)

        emergency_cost_per_mw = 75.0
        penalty = max(shortfall, 0.0) * emergency_cost_per_mw
        realized_total_cost = planned_cost + penalty

        return {
            "true_demand": true_demand,
            "predicted_demand": predicted_demand,
            "demand_error": predicted_demand - true_demand,
            "P_planned": P_planned,
            "P_ideal": P_ideal,
            "shortfall_mw": shortfall,
            "planned_cost": planned_cost,
            "ideal_cost": ideal_cost,
            "realized_cost": realized_total_cost,
            "excess_cost": realized_total_cost - ideal_cost,
        }
