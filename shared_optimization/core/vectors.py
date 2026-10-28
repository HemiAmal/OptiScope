"""
Vector operations and numerical validation routines.
"""

from typing import List, Sequence, Union
import numpy as np

Number = Union[int, float, np.number]
Vector = Union[List[float], np.ndarray]


def is_number(val: object) -> bool:
    """Check if a value is a real numerical scalar (excluding bool)."""
    return (isinstance(val, (int, float, np.number)) and not isinstance(val, bool))


def validate_vector(v: Sequence[Number], name: str = "vector") -> List[float]:
    """
    Validate that an input represents a non-empty sequence of real numbers.
    Accepts Python lists, tuples, and NumPy 1D ndarrays.

    Returns:
        List[float]: A validated new list of floats.
    """
    if isinstance(v, np.ndarray):
        if v.ndim != 1:
            raise ValueError(f"NumPy array {name} must be 1-dimensional, got shape {v.shape}.")
        if v.size == 0:
            raise ValueError(f"Vector {name} cannot be empty.")
        if not np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.bool_):
            raise TypeError(f"NumPy array {name} must have numeric dtype, got {v.dtype}.")
        return [float(x) for x in v]

    if not isinstance(v, (list, tuple)):
        raise TypeError(f"Expected {name} to be a list, tuple, or ndarray, got {type(v).__name__}.")

    if len(v) == 0:
        raise ValueError(f"Vector {name} cannot be empty.")

    validated: List[float] = []
    for i, x in enumerate(v):
        if not is_number(x):
            raise TypeError(
                f"Element {name}[{i}] must be an int or float, got {type(x).__name__} ({repr(x)})."
            )
        validated.append(float(x))

    return validated


def check_same_dimension(u: Sequence[Number], v: Sequence[Number], op_name: str = "operation") -> None:
    """Ensure two vectors have identical dimensions."""
    if len(u) != len(v):
        raise ValueError(
            f"Dimension mismatch for {op_name}: vector 1 has dimension {len(u)}, "
            f"but vector 2 has dimension {len(v)}."
        )


def vec_add(u: Sequence[Number], v: Sequence[Number]) -> List[float]:
    """Compute vector addition: w = u + v."""
    u_val = validate_vector(u, "u")
    v_val = validate_vector(v, "v")
    check_same_dimension(u_val, v_val, "vector addition")
    return [u_val[i] + v_val[i] for i in range(len(u_val))]


def vec_sub(u: Sequence[Number], v: Sequence[Number]) -> List[float]:
    """Compute vector subtraction: w = u - v."""
    u_val = validate_vector(u, "u")
    v_val = validate_vector(v, "v")
    check_same_dimension(u_val, v_val, "vector subtraction")
    return [u_val[i] - v_val[i] for i in range(len(u_val))]


def scalar_mul(c: Number, v: Sequence[Number]) -> List[float]:
    """Compute scalar-vector multiplication: w = c * v."""
    if not is_number(c):
        raise TypeError(f"Scalar multiplier must be int or float, got {type(c).__name__}.")
    v_val = validate_vector(v, "v")
    c_float = float(c)
    return [c_float * x for x in v_val]


def dot_product(u: Sequence[Number], v: Sequence[Number]) -> float:
    """Compute the standard inner (dot) product <u, v>."""
    u_val = validate_vector(u, "u")
    v_val = validate_vector(v, "v")
    check_same_dimension(u_val, v_val, "dot product")
    return float(np.dot(u_val, v_val))


def norm_l1(v: Sequence[Number]) -> float:
    """Compute the L1 (Manhattan) norm: ||v||_1 = sum |v_i|."""
    v_val = validate_vector(v, "v")
    return float(np.linalg.norm(v_val, ord=1))


def norm_l2(v: Sequence[Number]) -> float:
    """
    Compute the L2 (Euclidean) norm with Blue's numerical scaling.
    Prevents floating-point overflow when elements are extremely large (> 1e154)
    and underflow when elements are near machine precision.
    """
    v_val = validate_vector(v, "v")
    max_abs = norm_linf(v_val)
    if max_abs == 0.0:
        return 0.0
    arr = np.array(v_val, dtype=float) / max_abs
    return float(max_abs * np.sqrt(np.sum(arr * arr)))


def norm_linf(v: Sequence[Number]) -> float:
    """Compute the L_infinity (Maximum) norm: ||v||_inf = max |v_i|."""
    v_val = validate_vector(v, "v")
    return float(np.linalg.norm(v_val, ord=np.inf))


def normalize_vector(v: Sequence[Number], tol: float = 1e-15) -> List[float]:
    """Return unit vector u = v / ||v||_2."""
    v_val = validate_vector(v, "v")
    mag = norm_l2(v_val)
    if mag <= tol:
        raise ValueError(f"Cannot normalize zero or near-zero vector (norm = {mag}).")
    return [x / mag for x in v_val]
