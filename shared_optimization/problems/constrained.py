"""
Constrained optimization problem formulation and data structures:
    min_x f(x)
    subject to:
        h_i(x) = 0,   i = 1, ..., m   (Equality constraints)
        g_j(x) <= 0,  j = 1, ..., p   (Inequality constraints)
"""

from typing import Callable, List, Sequence, Tuple
from shared_optimization.core.vectors import validate_vector, dot_product, Number
from shared_optimization.core.matrices import mat_vec_mul, Matrix

ConstraintFunction = Callable[[Sequence[Number]], float]
ConstraintGradient = Callable[[Sequence[Number]], List[float]]


class ConstrainedProblem:
    """
    Encapsulates a constrained optimization problem.

    Attributes:
        f: Objective function R^n -> R.
        grad_f: Objective gradient function.
        eq_constraints: List of equality constraint functions h_i(x) = 0.
        eq_gradients: List of equality constraint gradient functions nabla h_i(x).
        ineq_constraints: List of inequality constraint functions g_j(x) <= 0.
        ineq_gradients: List of inequality constraint gradient functions nabla g_j(x).
    """
    def __init__(
        self,
        f: Callable[[Sequence[Number]], float],
        grad_f: Callable[[Sequence[Number]], List[float]],
        eq_constraints: Sequence[ConstraintFunction] = (),
        eq_gradients: Sequence[ConstraintGradient] = (),
        ineq_constraints: Sequence[ConstraintFunction] = (),
        ineq_gradients: Sequence[ConstraintGradient] = (),
        name: str = "Constrained Problem",
    ) -> None:
        self.f = f
        self.grad_f = grad_f
        self.eq_constraints = list(eq_constraints)
        self.eq_gradients = list(eq_gradients)
        self.ineq_constraints = list(ineq_constraints)
        self.ineq_gradients = list(ineq_gradients)
        self.name = name


def create_standard_equality_problem() -> ConstrainedProblem:
    """
    Standard equality constrained test problem:
        min (x_1 - 1)^2 + (x_2 - 2)^2
        s.t. x_1 + x_2 - 1 = 0

    Analytical Optimum:
        x* = [0.0, 1.0], lambda* = -2.0, f(x*) = 2.0
    """
    def f(x):
        return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2

    def grad_f(x):
        return [2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)]

    def h1(x):
        return x[0] + x[1] - 1.0

    def grad_h1(x):
        return [1.0, 1.0]

    return ConstrainedProblem(
        f=f,
        grad_f=grad_f,
        eq_constraints=[h1],
        eq_gradients=[grad_h1],
        name="Quadratic with Linear Equality (x1 + x2 = 1)",
    )


def create_standard_kkt_problem() -> ConstrainedProblem:
    """
    Standard inequality constrained test problem:
        min x_1^2 + x_2^2
        s.t. 1 - x_1 - x_2 <= 0  (i.e. x_1 + x_2 >= 1)

    Analytical Optimum:
        x* = [0.5, 0.5], mu* = 1.0, active constraint g_1(x*) = 0.
    """
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    def grad_f(x):
        return [2.0 * x[0], 2.0 * x[1]]

    def g1(x):
        return 1.0 - x[0] - x[1]

    def grad_g1(x):
        return [-1.0, -1.0]

    return ConstrainedProblem(
        f=f,
        grad_f=grad_f,
        ineq_constraints=[g1],
        ineq_gradients=[grad_g1],
        name="Circle with Halfspace Inequality (x1 + x2 >= 1)",
    )
