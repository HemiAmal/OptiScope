"""
Numerical optimization algorithms library.
"""

from shared_optimization.algorithms.line_search import (
    exact_line_search_quadratic,
    backtracking_line_search,
    armijo_condition,
)
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
from shared_optimization.algorithms.regression_optimizers import (
    train_linear_regression_gd,
    train_linear_regression_sgd,
    train_linear_regression_minibatch,
    train_logistic_regression,
)
from shared_optimization.algorithms.adaptive_optimizers import (
    adagrad_optimizer,
    rmsprop_optimizer,
    adam_optimizer,
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

__all__ = [
    "exact_line_search_quadratic",
    "backtracking_line_search",
    "armijo_condition",
    "gradient_descent_fixed",
    "gradient_descent_exact_quadratic",
    "gradient_descent_armijo",
    "momentum_gd",
    "nesterov_gd",
    "newton_method",
    "damped_newton_method",
    "verify_newton_one_step",
    "bfgs_optimizer",
    "train_linear_regression_gd",
    "train_linear_regression_sgd",
    "train_linear_regression_minibatch",
    "train_logistic_regression",
    "adagrad_optimizer",
    "rmsprop_optimizer",
    "adam_optimizer",
    "subgradient_method",
    "subgradient_lasso",
    "lagrange_multiplier_solver",
    "kkt_conditions_solver",
    "check_licq",
]
