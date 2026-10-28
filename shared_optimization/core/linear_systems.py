"""
Direct linear system solvers using Gaussian elimination with partial pivoting (GEPP),
back-substitution, Gauss-Jordan matrix inversion, and determinant computation.
"""

from typing import List, Sequence, Tuple, Union
from shared_optimization.core.vectors import validate_vector, Number, Vector
from shared_optimization.core.matrices import validate_matrix, matrix_shape, Matrix, identity_matrix

TOLERANCE_SINGULAR = 1e-14


class SingularMatrixError(ValueError):
    """Raised when an equation system or matrix inversion encounters a singular matrix."""
    pass


def solve_linear_system(
    A: Sequence[Sequence[Number]],
    b: Sequence[Number],
    tol: float = TOLERANCE_SINGULAR,
) -> List[float]:
    """
    Solve the linear system A * x = b for x using Gaussian Elimination with Partial Pivoting.

    Parameters:
        A: Square coefficient matrix of shape n x n.
        b: Right-hand side vector of length n.
        tol: Near-zero threshold for identifying singular matrices.

    Returns:
        List[float]: Solution vector x such that A * x = b.

    Raises:
        ValueError: If A is not square, dimensions don't match, or A is singular.
    """
    A_val = validate_matrix(A, "A")
    b_val = validate_vector(b, "b")

    n, cols = matrix_shape(A_val)
    if n != cols:
        raise ValueError(f"Coefficient matrix A must be square, got shape ({n}, {cols}).")
    if n != len(b_val):
        raise ValueError(
            f"Dimension mismatch: matrix A has {n} rows, but vector b has length {len(b_val)}."
        )

    # Make working copies of A and b
    U: Matrix = [list(row) for row in A_val]
    c: List[float] = list(b_val)

    # 1. Forward Elimination with Partial Pivoting
    for k in range(n - 1):
        # Find pivot row with maximum absolute entry in column k
        max_val = abs(U[k][k])
        pivot_row = k
        for p in range(k + 1, n):
            val = abs(U[p][k])
            if val > max_val:
                max_val = val
                pivot_row = p

        if max_val < tol:
            raise SingularMatrixError(
                f"Singular matrix encountered at column {k}: pivot magnitude {max_val:.2e} < {tol}."
            )

        # Swap rows if pivot_row != k
        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            c[k], c[pivot_row] = c[pivot_row], c[k]

        # Eliminate rows below pivot
        pivot = U[k][k]
        for i in range(k + 1, n):
            factor = U[i][k] / pivot
            if factor != 0.0:
                U[i][k] = 0.0  # Explicitly zero the eliminated entry
                for j in range(k + 1, n):
                    U[i][j] -= factor * U[k][j]
                c[i] -= factor * c[k]

    # Check the final diagonal element
    if abs(U[n - 1][n - 1]) < tol:
        raise SingularMatrixError(
            f"Singular matrix: final pivot U[{n-1}][{n-1}] has magnitude {abs(U[n-1][n-1]):.2e} < {tol}."
        )

    # 2. Back-Substitution
    x: List[float] = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = c[i]
        for j in range(i + 1, n):
            total -= U[i][j] * x[j]
        x[i] = total / U[i][i]

    return x


def invert_matrix(
    A: Sequence[Sequence[Number]],
    tol: float = TOLERANCE_SINGULAR,
) -> Matrix:
    """
    Compute the inverse A^{-1} of an n x n matrix using Gauss-Jordan Elimination with Partial Pivoting.

    Parameters:
        A: Square matrix of shape n x n.
        tol: Near-zero threshold for identifying non-invertible matrices.

    Returns:
        Matrix: Inverse matrix of shape n x n such that A * A^{-1} = I_n.

    Raises:
        ValueError: If A is non-square or singular.
    """
    A_val = validate_matrix(A, "A")
    n, cols = matrix_shape(A_val)
    if n != cols:
        raise ValueError(f"Matrix must be square to invert, got shape ({n}, {cols}).")

    # Construct augmented matrix [A | I_n]
    I = identity_matrix(n)
    aug: Matrix = [A_val[i] + I[i] for i in range(n)]

    # Gauss-Jordan Elimination
    for k in range(n):
        # Partial pivoting
        max_val = abs(aug[k][k])
        pivot_row = k
        for p in range(k + 1, n):
            val = abs(aug[p][k])
            if val > max_val:
                max_val = val
                pivot_row = p

        if max_val < tol:
            raise SingularMatrixError(
                f"Matrix is singular to working precision (pivot {max_val:.2e} < {tol}). Cannot invert."
            )

        if pivot_row != k:
            aug[k], aug[pivot_row] = aug[pivot_row], aug[k]

        # Normalize pivot row
        pivot = aug[k][k]
        for j in range(2 * n):
            aug[k][j] /= pivot

        # Eliminate all other rows in column k
        for i in range(n):
            if i != k:
                factor = aug[i][k]
                if factor != 0.0:
                    for j in range(2 * n):
                        aug[i][j] -= factor * aug[k][j]

    # Extract right half [I_n | A^{-1}]
    inv: Matrix = [[aug[i][j + n] for j in range(n)] for i in range(n)]
    return inv


def matrix_determinant(
    A: Sequence[Sequence[Number]],
    tol: float = TOLERANCE_SINGULAR,
) -> float:
    """
    Compute the determinant of an n x n matrix using Gaussian Elimination.

    Mathematical Definition:
        det(A) = (-1)^s * prod_{i=0}^{n-1} U_{ii}
    where s is the number of row swaps performed during partial pivoting.

    Parameters:
        A: Square matrix of shape n x n.

    Returns:
        float: Determinant of matrix A.
    """
    A_val = validate_matrix(A, "A")
    n, cols = matrix_shape(A_val)
    if n != cols:
        raise ValueError(f"Matrix must be square to compute determinant, got shape ({n}, {cols}).")

    U: Matrix = [list(row) for row in A_val]
    det = 1.0
    swaps = 0

    for k in range(n):
        max_val = abs(U[k][k])
        pivot_row = k
        for p in range(k + 1, n):
            val = abs(U[p][k])
            if val > max_val:
                max_val = val
                pivot_row = p

        if max_val < tol:
            return 0.0

        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            swaps += 1

        pivot = U[k][k]
        det *= pivot

        for i in range(k + 1, n):
            factor = U[i][k] / pivot
            if factor != 0.0:
                for j in range(k + 1, n):
                    U[i][j] -= factor * U[k][j]

    if swaps % 2 != 0:
        det = -det

    return det


def solve_linear_system_numpy(
    A: Sequence[Sequence[Number]],
    b: Sequence[Number],
) -> List[float]:
    """
    Solve A * x = b using NumPy's LAPACK linear algebra solver.

    Raises:
        SingularMatrixError: If A is singular.
    """
    import numpy as np
    A_val = validate_matrix(A, "A")
    b_val = validate_vector(b, "b")
    try:
        sol = np.linalg.solve(np.array(A_val, dtype=float), np.array(b_val, dtype=float))
        return sol.tolist()
    except np.linalg.LinAlgError as e:
        raise SingularMatrixError(f"NumPy LinAlgError: {e}")


def matrix_condition_number(A: Sequence[Sequence[Number]]) -> float:
    """Compute the 2-norm condition number kappa(A) = sigma_max / sigma_min using NumPy SVD."""
    import numpy as np
    A_val = validate_matrix(A, "A")
    return float(np.linalg.cond(np.array(A_val, dtype=float), p=2))

