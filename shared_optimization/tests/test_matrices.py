"""
Unit Tests for Matrix Module (core/matrices.py).
"""

import unittest
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


class TestMatrixOperations(unittest.TestCase):
    def test_identity_and_shape(self):
        I3 = identity_matrix(3)
        self.assertEqual(matrix_shape(I3), (3, 3))
        self.assertTrue(is_square(I3))
        self.assertTrue(is_symmetric(I3))
        self.assertEqual(I3[0], [1.0, 0.0, 0.0])
        self.assertEqual(I3[1], [0.0, 1.0, 0.0])
        self.assertEqual(I3[2], [0.0, 0.0, 1.0])

    def test_transpose(self):
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        At = transpose(A)
        self.assertEqual(matrix_shape(At), (3, 2))
        self.assertEqual(At, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_mat_add_sub_scalar(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        self.assertEqual(mat_add(A, B), [[6.0, 8.0], [10.0, 12.0]])
        self.assertEqual(mat_sub(B, A), [[4.0, 4.0], [4.0, 4.0]])
        self.assertEqual(mat_scalar_mul(2.0, A), [[2.0, 4.0], [6.0, 8.0]])

    def test_matrix_vector_multiplication(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        v = [5.0, 6.0]
        # [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
        self.assertEqual(mat_vec_mul(A, v), [17.0, 39.0])

    def test_matrix_matrix_multiplication(self):
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # 2x3
        B = [[7.0, 8.0], [9.0, 1.0], [2.0, 3.0]]  # 3x2
        # C = [[31, 19], [85, 55]]
        C = mat_mat_mul(A, B)
        self.assertEqual(C, [[31.0, 19.0], [85.0, 55.0]])

    def test_symmetry_check(self):
        S = [[1.0, 2.0], [2.0, 5.0]]
        self.assertTrue(is_symmetric(S))
        NS = [[1.0, 2.0], [3.0, 5.0]]
        self.assertFalse(is_symmetric(NS))

    def test_numpy_matrix_support(self):
        import numpy as np
        A_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        self.assertEqual(validate_matrix(A_np), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(matrix_shape(A_np), (2, 2))
        self.assertEqual(mat_vec_mul(A_np, [5.0, 6.0]), [17.0, 39.0])

    def test_dimension_mismatch_errors(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        v3 = [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            mat_vec_mul(A, v3)

        A_2x3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        B_2x2 = [[1.0, 2.0], [3.0, 4.0]]
        with self.assertRaises(ValueError):
            mat_mat_mul(A_2x3, B_2x2)


if __name__ == "__main__":
    unittest.main()
