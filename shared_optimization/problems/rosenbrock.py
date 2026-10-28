"""
Rosenbrock (banana valley) non-convex optimization benchmark:
    f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
Minimizer: x* = (1.0, 1.0) with f(x*) = 0.
"""

from typing import List, Sequence, Tuple
from shared_optimization.core.vectors import validate_vector, Number
from shared_optimization.core.matrices import Matrix

ROSENBROCK_OPTIMUM: List[float] = [1.0, 1.0]
ROSENBROCK_OPTIMUM_VALUE: float = 0.0


def _parse_2d_point(point: Sequence[Number]) -> Tuple[float, float]:
    v = validate_vector(point, "point")
    if len(v) != 2:
        raise ValueError(f"Rosenbrock problem expects 2D vector, got dimension {len(v)}.")
    return v[0], v[1]


def f_rosenbrock(point: Sequence[Number], a: float = 1.0, b: float = 100.0) -> float:
    """Evaluate Rosenbrock function: (a - x)^2 + b * (y - x^2)^2."""
    x, y = _parse_2d_point(point)
    term1 = a - x
    term2 = y - x * x
    return term1 * term1 + b * term2 * term2


def grad_rosenbrock(point: Sequence[Number], a: float = 1.0, b: float = 100.0) -> List[float]:
    """Compute analytical gradient."""
    x, y = _parse_2d_point(point)
    dx = -2.0 * (a - x) - 4.0 * b * x * (y - x * x)
    dy = 2.0 * b * (y - x * x)
    return [dx, dy]


def hess_rosenbrock(point: Sequence[Number], a: float = 1.0, b: float = 100.0) -> Matrix:
    """Compute analytical Hessian matrix."""
    x, y = _parse_2d_point(point)
    dxx = 2.0 - 4.0 * b * y + 12.0 * b * x * x
    dxy = -4.0 * b * x
    dyy = 2.0 * b
    return [
        [dxx, dxy],
        [dxy, dyy],
    ]


class RosenbrockProblem:
    """Encapsulates the 2D Rosenbrock benchmark problem."""
    def __init__(self, a: float = 1.0, b: float = 100.0) -> None:
        self.a = a
        self.b = b
        self.dimension = 2
        self.optimum = [a, a * a]
        self.optimum_value = 0.0
        self.name = "Rosenbrock Banana Function"

    def objective(self, point: Sequence[Number]) -> float:
        return f_rosenbrock(point, a=self.a, b=self.b)

    def gradient(self, point: Sequence[Number]) -> List[float]:
        return grad_rosenbrock(point, a=self.a, b=self.b)

    def hessian(self, point: Sequence[Number]) -> Matrix:
        return hess_rosenbrock(point, a=self.a, b=self.b)
