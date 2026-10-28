"""
Convergence criteria and optimization result containers.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from shared_optimization.core.vectors import norm_l2, vec_sub, Number


@dataclass
class IterationRecord:
    """Snapshot of a single optimizer iteration."""
    iteration: int
    x: List[float]
    fun: float
    grad_norm: float
    step_size: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """
    Standard result returned by numerical optimization algorithms.

    Attributes:
        x: Final parameter vector.
        fun: Final objective function value.
        grad_norm: Final gradient L2 norm.
        iterations: Number of optimizer updates completed.
        func_evals: Total scalar objective function evaluations.
        grad_evals: Total gradient evaluations.
        hess_evals: Total Hessian evaluations.
        converged: Boolean indicating whether stopping criteria were satisfied.
        status_message: Explanation of termination (e.g. 'Converged: gradient norm below tolerance').
        history: Sequence of IterationRecord objects capturing the trajectory.
        execution_time: Measured runtime in seconds.
    """
    x: List[float]
    fun: float
    grad_norm: float
    iterations: int
    func_evals: int
    grad_evals: int
    hess_evals: int
    converged: bool
    status_message: str
    history: List[IterationRecord] = field(default_factory=list)
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result summary to a dictionary."""
        return {
            "x": self.x,
            "fun": self.fun,
            "grad_norm": self.grad_norm,
            "iterations": self.iterations,
            "func_evals": self.func_evals,
            "grad_evals": self.grad_evals,
            "hess_evals": self.hess_evals,
            "converged": self.converged,
            "status_message": self.status_message,
            "execution_time": self.execution_time,
        }


@dataclass
class ConvergenceCriterion:
    """
    Encapsulates stopping tolerances and limits.

    Attributes:
        grad_tol: Gradient L2 norm threshold (default: 1e-5).
        step_tol: Step change ||x_{k+1} - x_k||_2 threshold (default: 1e-8).
        obj_tol: Relative objective change threshold (default: 1e-12).
        max_iters: Maximum iterations allowed (default: 1000).
    """
    grad_tol: float = 1e-5
    step_tol: float = 1e-8
    obj_tol: float = 1e-12
    max_iters: int = 1000


def check_finite(x: Sequence[Number], name: str = "vector") -> None:
    """Check that all vector components are finite (neither NaN nor Inf)."""
    for i, val in enumerate(x):
        if math.isnan(val) or math.isinf(val):
            raise FloatingPointError(f"Non-finite value encountered in {name}[{i}]: {val}.")


def check_convergence(
    grad_norm: float,
    step_norm: Optional[float] = None,
    delta_f: Optional[float] = None,
    f_current: Optional[float] = None,
    criterion: Optional[ConvergenceCriterion] = None,
) -> Optional[str]:
    """
    Check if stopping criteria have been satisfied.

    Returns:
        Optional[str]: Success message if converged, None otherwise.
    """
    crit = criterion if criterion is not None else ConvergenceCriterion()

    # 1. First-order stationarity: ||nabla f(x)|| < grad_tol
    if grad_norm < crit.grad_tol:
        return f"Converged: gradient norm ||nabla f|| = {grad_norm:.2e} < {crit.grad_tol:.2e}"

    # 2. Step change: ||x_{k+1} - x_k|| < step_tol
    if step_norm is not None and step_norm < crit.step_tol:
        return f"Converged: step size ||dx|| = {step_norm:.2e} < {crit.step_tol:.2e}"

    # 3. Objective stagnation: |f_{k+1} - f_k| / (|f_k| + 1) < obj_tol
    if delta_f is not None and f_current is not None:
        rel_change = abs(delta_f) / (abs(f_current) + 1.0)
        if rel_change < crit.obj_tol:
            return f"Converged: relative function change = {rel_change:.2e} < {crit.obj_tol:.2e}"

    return None
