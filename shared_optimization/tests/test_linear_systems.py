"""
Unit Tests for Linear System Solver (core/linear_systems.py).
"""

import unittest
from shared_optimization.core.linear_systems import (
    solve_linear_system,
    invert_matrix,
    matrix_determinant,
    SingularMatrixError,
)
from shared_optimization.core.matrices import mat_mat_mul, mat_vec_mul, identity_matrix


class TestLinearSystems(unittest.TestCase):
    def test_solve_2x2_system(self):
        # 2x + y = 5
        # x + 3y = 5
        # Solution: x = 2, y = 1
        A = [[2.0, 1.0], [1.0, 3.0]]
        b = [5.0, 5.0]
        x = solve_linear_system(A, b)
        self.assertAlmostEqual(x[0], 2.0)
        self.assertAlmostEqual(x[1], 1.0)

    def test_solve_system_with_zero_pivot(self):
        # System where A[0][0] is 0, requiring partial pivoting row swap:
        # 0*x + 2*y + 1*z = 4
        # 1*x - 1*y + 2*z = 5
        # 2*x + 1*y - 1*z = 1
        A = [
            [0.0, 2.0, 1.0],
            [1.0, -1.0, 2.0],
            [2.0, 1.0, -1.0],
        ]
        b = [4.0, 5.0, 1.0]
        x = solve_linear_system(A, b)

        # Check A * x == b
        Ax = mat_vec_mul(A, x)
        for i in range(3):
            self.assertAlmostEqual(Ax[i], b[i], places=7)

    def test_singular_matrix_error(self):
        # Dependent rows: row 1 is 2 * row 0
        A = [[1.0, 2.0], [2.0, 4.0]]
        b = [3.0, 6.0]
        with self.assertRaises(SingularMatrixError):
            solve_linear_system(A, b)

        with self.assertRaises(SingularMatrixError):
            invert_matrix(A)

    def test_invert_matrix(self):
        A = [[4.0, 7.0], [2.0, 6.0]]
        A_inv = invert_matrix(A)

        # A * A_inv must equal I_2
        prod = mat_mat_mul(A, A_inv)
        I = identity_matrix(2)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(prod[i][j], I[i][j], places=7)

    def test_matrix_determinant(self):
        # det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2.0
        A = [[1.0, 2.0], [3.0, 4.0]]
        self.assertAlmostEqual(matrix_determinant(A), -2.0)

        # det(I_3) = 1.0
        I3 = identity_matrix(3)
        self.assertAlmostEqual(matrix_determinant(I3), 1.0)

        # det(singular) = 0.0
        A_sing = [[1.0, 2.0], [2.0, 4.0]]
        self.assertAlmostEqual(matrix_determinant(A_sing), 0.0)


if __name__ == "__main__":
    unittest.main()
