import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
from shared_optimization.core.convergence import ConvergenceCriterion


class IllConditionedQuadratic:
    def __init__(self, cond_ratio: float = 10.0):
        self.Q = np.array([[2.0 * cond_ratio, 0.0], [0.0, 2.0]], dtype=float)
        self.b = np.array([0.0, 0.0], dtype=float)
        self.x_star = np.array([0.0, 0.0], dtype=float)
        self.condition_number = float(np.linalg.cond(self.Q))

    def f(self, x: List[float]) -> float:
        xa = np.array(x, dtype=float)
        return float(0.5 * xa.T @ self.Q @ xa - self.b.T @ xa)

    def grad(self, x: List[float]) -> List[float]:
        xa = np.array(x, dtype=float)
        return (self.Q @ xa - self.b).tolist()

    def hess(self, x: List[float]) -> List[List[float]]:
        return self.Q.tolist()

    def run_all_quadratic_solvers(
        self,
        x0: List[float] = [-4.0, 3.0],
        max_iters: int = 200,
    ) -> Dict[str, Any]:
        results = {}
        crit = ConvergenceCriterion(max_iters=max_iters, grad_tol=1e-5, step_tol=1e-6)

        res_gd = gradient_descent_fixed(self.f, self.grad, x0, learning_rate=0.04, criterion=crit)
        results["Fixed GD"] = {"alg_num": 1, "result": res_gd}

        res_exact = gradient_descent_exact_quadratic(x0, self.Q, criterion=crit)
        results["Exact Line Search GD"] = {"alg_num": 2, "result": res_exact}

        res_armijo = gradient_descent_armijo(self.f, self.grad, x0, alpha_init=1.0, rho=0.5, c1=1e-4, criterion=crit)
        results["Armijo GD"] = {"alg_num": 5, "result": res_armijo}

        res_mom = momentum_gd(self.f, self.grad, x0, learning_rate=0.04, beta=0.85, criterion=crit)
        results["Polyak Momentum"] = {"alg_num": 6, "result": res_mom}

        res_nes = nesterov_gd(self.f, self.grad, x0, learning_rate=0.04, beta=0.85, criterion=crit)
        results["Nesterov GD"] = {"alg_num": 7, "result": res_nes}

        res_newton = newton_method(self.f, self.grad, self.hess, x0, criterion=crit)
        results["Newton Method"] = {"alg_num": 17, "result": res_newton}

        res_damped = damped_newton_method(self.f, self.grad, self.hess, x0, alpha_init=1.0, rho=0.5, c1=1e-4, criterion=crit)
        results["Damped Newton"] = {"alg_num": 18, "result": res_damped}

        res_bfgs = bfgs_optimizer(self.f, self.grad, x0, alpha_init=1.0, rho_bt=0.5, c1=1e-4, criterion=crit)
        results["BFGS"] = {"alg_num": 20, "result": res_bfgs}

        return results

    def run_newton_one_step_demonstration(self, x0: List[float] = [10.0, -15.0]) -> Dict[str, Any]:
        c_vec = (-self.b).tolist()
        return verify_newton_one_step(self.Q, c_vec, x0)
