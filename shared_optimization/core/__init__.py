"""
Core numerical linear algebra and calculus utilities.
"""

from shared_optimization.core.vectors import (
    is_number,
    validate_vector,
    vec_add,
    vec_sub,
    scalar_mul,
    dot_product,
    norm_l1,
    norm_l2,
    norm_linf,
)
from shared_optimization.core.matrices import (
    validate_matrix,
    matrix_shape,
    transpose,
    identity_matrix,
    mat_add,
    mat_sub,
    mat_scalar_mul,
    mat_vec_mul,
    mat_mat_mul,
    is_square,
    is_symmetric,
)
from shared_optimization.core.linear_systems import (
    solve_linear_system,
    solve_linear_system_numpy,
    matrix_condition_number,
    invert_matrix,
    matrix_determinant,
    SingularMatrixError,
)
from shared_optimization.core.derivatives import (
    finite_difference_gradient,
    finite_difference_hessian,
    finite_difference_hessian_from_grad,
    verify_gradient,
    verify_hessian,
)
from shared_optimization.core.convergence import (
    OptimizationResult,
    ConvergenceCriterion,
    check_convergence,
)

__all__ = [
    "is_number",
    "validate_vector",
    "vec_add",
    "vec_sub",
    "scalar_mul",
    "dot_product",
    "norm_l1",
    "norm_l2",
    "norm_linf",
    "validate_matrix",
    "matrix_shape",
    "transpose",
    "identity_matrix",
    "mat_add",
    "mat_sub",
    "mat_scalar_mul",
    "mat_vec_mul",
    "mat_mat_mul",
    "is_square",
    "is_symmetric",
    "solve_linear_system",
    "solve_linear_system_numpy",
    "matrix_condition_number",
    "invert_matrix",
    "matrix_determinant",
    "SingularMatrixError",
    "finite_difference_gradient",
    "finite_difference_hessian",
    "finite_difference_hessian_from_grad",
    "verify_gradient",
    "verify_hessian",
    "OptimizationResult",
    "ConvergenceCriterion",
    "check_convergence",
]
