"""
Unit Tests for Vector Module (core/vectors.py).
"""

import unittest
from shared_optimization.core.vectors import (
    validate_vector,
    vec_add,
    vec_sub,
    scalar_mul,
    dot_product,
    norm_l1,
    norm_l2,
    norm_linf,
    normalize_vector,
)


class TestVectorOperations(unittest.TestCase):
    def test_vector_addition_and_subtraction(self):
        u = [1.0, 2.0, 3.0]
        v = [4.0, 5.0, 6.0]
        self.assertEqual(vec_add(u, v), [5.0, 7.0, 9.0])
        self.assertEqual(vec_sub(u, v), [-3.0, -3.0, -3.0])

    def test_scalar_multiplication(self):
        v = [2.0, -4.0, 0.0]
        self.assertEqual(scalar_mul(2.5, v), [5.0, -10.0, 0.0])
        self.assertEqual(scalar_mul(0.0, v), [0.0, 0.0, 0.0])

    def test_dot_product(self):
        u = [1.0, 3.0, -5.0]
        v = [4.0, -2.0, -1.0]
        # 1*4 + 3*(-2) + (-5)*(-1) = 4 - 6 + 5 = 3
        self.assertAlmostEqual(dot_product(u, v), 3.0)

        # Orthogonal vectors: [1, 0] . [0, 1] = 0
        self.assertAlmostEqual(dot_product([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_norms_hand_calculated(self):
        v = [3.0, -4.0]
        self.assertAlmostEqual(norm_l1(v), 7.0)
        self.assertAlmostEqual(norm_l2(v), 5.0)
        self.assertAlmostEqual(norm_linf(v), 4.0)

        # Zero vector
        zero = [0.0, 0.0, 0.0]
        self.assertEqual(norm_l1(zero), 0.0)
        self.assertEqual(norm_l2(zero), 0.0)
        self.assertEqual(norm_linf(zero), 0.0)

    def test_scaled_norm_l2_extreme_values(self):
        # Extreme large values that would overflow if squared naively (> 1e154)
        large_v = [1e200, 1e200]
        # Euclidean norm: sqrt(2 * (1e200)^2) = 1e200 * sqrt(2)
        norm_val = norm_l2(large_v)
        self.assertAlmostEqual(norm_val / 1e200, 2.0 ** 0.5, places=10)

    def test_normalization(self):
        v = [3.0, 4.0]
        u = normalize_vector(v)
        self.assertAlmostEqual(u[0], 0.6)
        self.assertAlmostEqual(u[1], 0.8)
        self.assertAlmostEqual(norm_l2(u), 1.0)

        with self.assertRaises(ValueError):
            normalize_vector([0.0, 0.0])

    def test_numpy_array_support(self):
        import numpy as np
        v_np = np.array([3.0, -4.0], dtype=float)
        self.assertEqual(validate_vector(v_np), [3.0, -4.0])
        self.assertAlmostEqual(norm_l2(v_np), 5.0)
        self.assertAlmostEqual(dot_product(v_np, [1.0, 2.0]), -5.0)

    def test_invalid_dimensions_and_types(self):
        with self.assertRaises(ValueError):
            vec_add([1.0, 2.0], [1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            validate_vector([])
        with self.assertRaises(TypeError):
            validate_vector([1.0, "two"])
        with self.assertRaises(TypeError):
            validate_vector([1.0, True])


if __name__ == "__main__":
    unittest.main()
