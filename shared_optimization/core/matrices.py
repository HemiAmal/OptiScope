"""
Matrix operations and linear algebra utility routines.
"""

from typing import List, Sequence, Tuple, Union
import numpy as np
from shared_optimization.core.vectors import is_number, validate_vector, Number

Matrix = Union[List[List[float]], np.ndarray]


def validate_matrix(M: Sequence[Sequence[Number]], name: str = "matrix") -> List[List[float]]:
    """
    Validate that an input represents a rectangular 2D array of real numbers.
    Accepts nested lists, tuples, and 2D NumPy ndarrays.

    Returns:
        List[List[float]]: A newly allocated rectangular matrix of floats.
    """
    if isinstance(M, np.ndarray):
        if M.ndim != 2:
            raise ValueError(f"NumPy array {name} must be 2-dimensional, got shape {M.shape}.")
        if M.size == 0 or M.shape[0] == 0 or M.shape[1] == 0:
            raise ValueError(f"Matrix {name} must have non-zero dimensions, got shape {M.shape}.")
        if not np.issubdtype(M.dtype, np.number) or np.issubdtype(M.dtype, np.bool_):
            raise TypeError(f"NumPy array {name} must have numeric dtype, got {M.dtype}.")
        return [[float(x) for x in row] for row in M]

    if not isinstance(M, (list, tuple)):
        raise TypeError(f"Expected {name} to be a list, tuple, or 2D ndarray, got {type(M).__name__}.")

    m = len(M)
    if m == 0:
        raise ValueError(f"Matrix {name} must have at least one row.")

    first_row = M[0]
    if not isinstance(first_row, (list, tuple)):
        raise TypeError(f"Row 0 of matrix {name} must be a list or tuple, got {type(first_row).__name__}.")

    n = len(first_row)
    if n == 0:
        raise ValueError(f"Matrix {name} must have at least one column (row 0 is empty).")

    validated: List[List[float]] = []
    for i in range(m):
        row = M[i]
        if not isinstance(row, (list, tuple)):
            raise TypeError(f"Row {i} of matrix {name} must be a list or tuple, got {type(row).__name__}.")

        if len(row) != n:
            raise ValueError(
                f"Irregular matrix dimensions for {name}: row 0 has {n} columns, "
                f"but row {i} has {len(row)} columns."
            )

        validated_row: List[float] = []
        for j in range(n):
            val = row[j]
            if not is_number(val):
                raise TypeError(
                    f"Entry {name}[{i}][{j}] must be an int or float, got {type(val).__name__} ({repr(val)})."
                )
            validated_row.append(float(val))
        validated.append(validated_row)

    return validated


def matrix_shape(M: Matrix) -> Tuple[int, int]:
    """Return the dimensions (rows, cols) of a matrix."""
    if isinstance(M, np.ndarray):
        return M.shape[0], M.shape[1]
    return len(M), len(M[0])


def is_square(M: Sequence[Sequence[Number]]) -> bool:
    """Check if a matrix is square (m == n)."""
    M_val = validate_matrix(M, "M")
    m, n = matrix_shape(M_val)
    return m == n


def is_symmetric(M: Sequence[Sequence[Number]], tol: float = 1e-8) -> bool:
    """Check if a square matrix is symmetric: A_{ij} == A_{ji} within tolerance."""
    M_val = validate_matrix(M, "M")
    m, n = matrix_shape(M_val)
    if m != n:
        return False

    arr = np.array(M_val, dtype=float)
    return bool(np.allclose(arr, arr.T, atol=tol))


def transpose(A: Sequence[Sequence[Number]]) -> List[List[float]]:
    """Compute transpose A^T."""
    A_val = validate_matrix(A, "A")
    arr = np.array(A_val, dtype=float)
    return arr.T.tolist()


def identity_matrix(n: int) -> List[List[float]]:
    """Construct n x n identity matrix I_n."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"Dimension n must be an integer, got {type(n).__name__}.")
    if n < 1:
        raise ValueError(f"Dimension n must be at least 1, got {n}.")
    return np.eye(n, dtype=float).tolist()


def mat_add(
    A: Sequence[Sequence[Number]], B: Sequence[Sequence[Number]]
) -> List[List[float]]:
    """Compute matrix addition: C = A + B."""
    A_val = validate_matrix(A, "A")
    B_val = validate_matrix(B, "B")
    m_A, n_A = matrix_shape(A_val)
    m_B, n_B = matrix_shape(B_val)

    if (m_A, n_A) != (m_B, n_B):
        raise ValueError(f"Shape mismatch for matrix addition: ({m_A}, {n_A}) != ({m_B}, {n_B}).")

    return (np.array(A_val, dtype=float) + np.array(B_val, dtype=float)).tolist()


def mat_sub(
    A: Sequence[Sequence[Number]], B: Sequence[Sequence[Number]]
) -> List[List[float]]:
    """Compute matrix subtraction: C = A - B."""
    A_val = validate_matrix(A, "A")
    B_val = validate_matrix(B, "B")
    m_A, n_A = matrix_shape(A_val)
    m_B, n_B = matrix_shape(B_val)

    if (m_A, n_A) != (m_B, n_B):
        raise ValueError(f"Shape mismatch for matrix subtraction: ({m_A}, {n_A}) != ({m_B}, {n_B}).")

    return (np.array(A_val, dtype=float) - np.array(B_val, dtype=float)).tolist()


def mat_scalar_mul(c: Number, A: Sequence[Sequence[Number]]) -> List[List[float]]:
    """Compute scalar-matrix multiplication: C = c * A."""
    if not is_number(c):
        raise TypeError(f"Scalar multiplier must be int or float, got {type(c).__name__}.")
    A_val = validate_matrix(A, "A")
    return (float(c) * np.array(A_val, dtype=float)).tolist()


def mat_vec_mul(A: Sequence[Sequence[Number]], v: Sequence[Number]) -> List[float]:
    """Compute matrix-vector multiplication: w = A * v."""
    A_val = validate_matrix(A, "A")
    v_val = validate_vector(v, "v")

    m, n = matrix_shape(A_val)
    if n != len(v_val):
        raise ValueError(f"Dimension mismatch in mat_vec_mul: matrix columns {n} != vector length {len(v_val)}.")

    return np.dot(A_val, v_val).tolist()


def mat_mat_mul(
    A: Sequence[Sequence[Number]], B: Sequence[Sequence[Number]]
) -> List[List[float]]:
    """Compute matrix-matrix multiplication: C = A * B."""
    A_val = validate_matrix(A, "A")
    B_val = validate_matrix(B, "B")

    m, n_A = matrix_shape(A_val)
    n_B, p = matrix_shape(B_val)

    if n_A != n_B:
        raise ValueError(f"Dimension mismatch in mat_mat_mul: matrix A cols {n_A} != matrix B rows {n_B}.")

    return np.matmul(A_val, B_val).tolist()
