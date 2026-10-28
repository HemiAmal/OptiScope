"""
Convex quadratic objective function, gradient, and Hessian benchmarks:
    f(x, y) = 0.5 * (x^2 + 100 * y^2)
"""

from typing import List, Sequence, Tuple
from shared_optimization.core.vectors import validate_vector, Number
from shared_optimization.core.matrices import Matrix

KNOWN_OPTIMUM: List[float] = [0.0, 0.0]
KNOWN_OPTIMUM_VALUE: float = 0.0
QUADRATIC_OPTIMUM = KNOWN_OPTIMUM
QUADRATIC_OPTIMUM_VALUE = KNOWN_OPTIMUM_VALUE


def _parse_2d_point(point: Sequence[Number]) -> Tuple[float, float]:
    v = validate_vector(point, "point")
    if len(v) != 2:
        raise ValueError(f"Quadratic problem expects 2D vector, got dimension {len(v)}.")
    return v[0], v[1]


def f_quadratic(point: Sequence[Number]) -> float:
    """Evaluate f(x, y) = 0.5 * (x^2 + 100 * y^2)."""
    x, y = _parse_2d_point(point)
    return 0.5 * (x * x + 100.0 * y * y)


def grad_quadratic(point: Sequence[Number]) -> List[float]:
    """Compute analytical gradient: [x, 100 * y]."""
    x, y = _parse_2d_point(point)
    return [x, 100.0 * y]


def hess_quadratic(point: Sequence[Number] = (0.0, 0.0)) -> Matrix:
    """Compute analytical Hessian: [[1, 0], [0, 100]]."""
    _parse_2d_point(point)
    return [
        [1.0, 0.0],
        [0.0, 100.0],
    ]


class QuadraticProblem:
    """Encapsulates the 2D ill-conditioned quadratic test problem."""
    def __init__(self) -> None:
        self.dimension = 2
        self.optimum = [0.0, 0.0]
        self.optimum_value = 0.0
        self.condition_number = 100.0
        self.name = "Ill-conditioned Quadratic (kappa=100)"

    def objective(self, point: Sequence[Number]) -> float:
        return f_quadratic(point)

    def gradient(self, point: Sequence[Number]) -> List[float]:
        return grad_quadratic(point)

    def hessian(self, point: Sequence[Number] = (0.0, 0.0)) -> Matrix:
        return hess_quadratic(point)
