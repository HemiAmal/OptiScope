import sys
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared_optimization.algorithms.gradient_descent import (
    gradient_descent_fixed,
    gradient_descent_armijo,
)
from shared_optimization.algorithms.momentum import momentum_gd
from shared_optimization.algorithms.nesterov import nesterov_gd
from shared_optimization.algorithms.newton import (
    newton_method,
    damped_newton_method,
)
from shared_optimization.algorithms.bfgs import bfgs_optimizer
from shared_optimization.core.convergence import ConvergenceCriterion


class RosenbrockCalibration:
    def __init__(self):
        self.x_star = [1.0, 1.0]

    def f(self, x: List[float]) -> float:
        x1, x2 = float(x[0]), float(x[1])
        return (1.0 - x1) ** 2 + 100.0 * (x2 - x1 ** 2) ** 2

    def grad(self, x: List[float]) -> List[float]:
        x1, x2 = float(x[0]), float(x[1])
        g1 = -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1 ** 2)
        g2 = 200.0 * (x2 - x1 ** 2)
        return [g1, g2]

    def hess(self, x: List[float]) -> List[List[float]]:
        x1, x2 = float(x[0]), float(x[1])
        h11 = 2.0 - 400.0 * (x2 - 3.0 * x1 ** 2)
        h12 = -400.0 * x1
        h21 = -400.0 * x1
        h22 = 200.0
        return [[h11, h12], [h21, h22]]

    def run_all_rosenbrock_solvers(
        self,
        x0: List[float] = [-1.2, 1.0],
        max_iters: int = 500,
    ) -> Dict[str, Any]:
        results = {}
        crit = ConvergenceCriterion(max_iters=max_iters, grad_tol=1e-5, step_tol=1e-7)

        res_gd = gradient_descent_fixed(self.f, self.grad, x0, learning_rate=0.001, criterion=crit)
        results["Fixed GD"] = {"alg_num": 1, "result": res_gd}

        res_armijo = gradient_descent_armijo(self.f, self.grad, x0, alpha_init=1.0, rho=0.5, c1=1e-4, criterion=crit)
        results["Armijo GD"] = {"alg_num": 5, "result": res_armijo}

        res_mom = momentum_gd(self.f, self.grad, x0, learning_rate=0.001, beta=0.9, criterion=crit)
        results["Polyak Momentum"] = {"alg_num": 6, "result": res_mom}

        res_nes = nesterov_gd(self.f, self.grad, x0, learning_rate=0.001, beta=0.9, criterion=crit)
        results["Nesterov GD"] = {"alg_num": 7, "result": res_nes}

        res_newton = newton_method(self.f, self.grad, self.hess, x0, criterion=crit)
        results["Newton Method"] = {"alg_num": 17, "result": res_newton}

        res_damped = damped_newton_method(self.f, self.grad, self.hess, x0, alpha_init=1.0, rho=0.5, c1=1e-4, criterion=crit)
        results["Damped Newton"] = {"alg_num": 18, "result": res_damped}

        res_bfgs = bfgs_optimizer(self.f, self.grad, x0, alpha_init=1.0, rho_bt=0.5, c1=1e-4, criterion=crit)
        results["BFGS"] = {"alg_num": 20, "result": res_bfgs}

        return results
