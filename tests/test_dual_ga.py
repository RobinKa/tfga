import unittest as ut
import numpy as np
import tensorflow as tf

from tfga import GeometricAlgebra

dual_metric = [0]
dual_bases = ["0"]
dual_blades = ["", "0"]
dual_blade_degrees = [len(blade) for blade in dual_blades]


class TestDualGeometricAlgebraMultiply(ut.TestCase):
    def test_mul_mv_mv(self):
        ga = GeometricAlgebra(metric=dual_metric)

        zero = ga.zeros([], kind="scalar")
        one = ga.ones([], kind="scalar")
        eps = ga.ones([], kind="pseudoscalar")
        ten = ga.fill([], fill_value=10.0, kind="scalar")

        self.assertEqual(eps * eps, zero)
        self.assertEqual(one * one, one)
        self.assertEqual(zero * one, zero)
        self.assertEqual(one * zero, zero)
        self.assertEqual(one * eps, eps)
        self.assertEqual(eps * one, eps)
        self.assertEqual(zero * zero, zero)
        self.assertEqual(ten * zero, zero)
        self.assertEqual(zero * ten, zero)
        self.assertEqual((ten * eps) * eps, zero)
        self.assertEqual(ten * one, ten)
        self.assertEqual(one * ten, ten)

    def test_mul_py_mv(self):
        ga = GeometricAlgebra(metric=dual_metric)

        zero = ga.zeros([], kind="scalar")
        zero_py = 0.0
        one = ga.ones([], kind="scalar")
        one_py = 1.0
        eps = ga.ones([], kind="pseudoscalar")
        ten = ga.fill([], fill_value=10.0, kind="scalar")
        ten_py = 10.0

        self.assertEqual(one * one_py, one)
        self.assertEqual(one_py * one, one)
        self.assertEqual(zero * one_py, zero)
        self.assertEqual(one_py * zero, zero)
        self.assertEqual(zero_py * one, zero)
        self.assertEqual(one * zero_py, zero)
        self.assertEqual(one_py * eps, eps)
        self.assertEqual(eps * one_py, eps)
        self.assertEqual(zero_py * zero, zero)
        self.assertEqual(zero * zero_py, zero)
        self.assertEqual(ten_py * zero, zero)
        self.assertEqual(zero * ten_py, zero)
        self.assertEqual(ten * zero_py, zero)
        self.assertEqual(zero_py * ten, zero)
        self.assertEqual((ten_py * eps) * eps, zero)
        self.assertEqual(ten_py * one, ten)
        self.assertEqual(one * ten_py, ten)
        self.assertEqual(ten * one_py, ten)
        self.assertEqual(one_py * ten, ten)

    def test_mul_tf_mv(self):
        ga = GeometricAlgebra(metric=dual_metric)

        zero = ga.zeros([], kind="scalar")
        zero_tf = ga.as_mv(tf.convert_to_tensor(0.0, dtype=tf.float32))
        one = ga.ones([], kind="scalar")
        one_tf = ga.as_mv(tf.convert_to_tensor(1.0, dtype=tf.float32))
        eps = ga.ones([], kind="pseudoscalar")
        ten = ga.fill([], fill_value=10.0, kind="scalar")
        ten_tf = ga.as_mv(tf.convert_to_tensor([10.0], dtype=tf.float32))

        self.assertEqual(one * one_tf, one)
        self.assertEqual(one_tf * one, one)
        self.assertEqual(zero * one_tf, zero)
        self.assertEqual(one_tf * zero, zero)
        self.assertEqual(zero_tf * one, zero)
        self.assertEqual(one * zero_tf, zero)
        self.assertEqual(one_tf * eps, eps)
        self.assertEqual(eps * one_tf, eps)
        self.assertEqual(zero_tf * zero, zero)
        self.assertEqual(zero * zero_tf, zero)
        self.assertEqual(ten_tf * zero, zero)
        self.assertEqual(zero * ten_tf, zero)
        self.assertEqual(ten * zero_tf, zero)
        self.assertEqual(zero_tf * ten, zero)
        self.assertEqual((ten_tf * eps) * eps, zero)
        self.assertEqual(ten_tf * one, ten)
        self.assertEqual(one * ten_tf, ten)
        self.assertEqual(ten * one_tf, ten)
        self.assertEqual(one_tf * ten, ten)


class TestDualGeometricAlgebraMisc(ut.TestCase):
    def assertAllElementsEqualTo(self, tensor, value):
        return tf.reduce_all(tensor == value)

    def test_auto_diff_square(self):
        """Test automatic differentiation using
        dual numbers for the square function.
        f(x) = x^2
        f'(x) = d/dx f(x) = 2x
        """
        ga = GeometricAlgebra(metric=dual_metric)

        one = ga.ones([], kind="scalar")
        five = 5.0 * ga.ones([], kind="scalar")
        eps = ga.ones([], kind="pseudoscalar")

        x = one + eps
        self.assertEqual(x[""], one)
        self.assertEqual(x["0"], eps)
        self.assertEqual(x.scalar, 1.0)
        self.assertEqual(x.get_part_mv("pseudoscalar"), eps)
        self.assertEqual(x.get_part("scalar"), 1.0)
        self.assertEqual(x.get_part("pseudoscalar"), 1.0)

        # f(1) = 1^2 = 1, f'(1) = 2
        x_squared = x * x
        self.assertEqual(x_squared.scalar, 1.0)
        self.assertEqual(x_squared["0"], 2.0 * eps)

        y = five + eps
        self.assertEqual(y[""], five)
        self.assertEqual(y["0"], eps)
        self.assertEqual(y.scalar, 5.0)
        self.assertEqual(y.get_part_mv("pseudoscalar"), eps)
        self.assertEqual(y.get_part("scalar"), 5.0)
        self.assertEqual(y.get_part("pseudoscalar"), 1.0)

        # f(5) = 5^2 = 25, f'(5) = 10
        y_squared = y * y
        self.assertEqual(y_squared.scalar, 25.0)
        self.assertEqual(y_squared["0"], eps * 10.0)

    def test_batched_auto_diff_square(self):
        """Test automatic differentiation using
        dual numbers for the square function.
        Use batch with identical elements.
        f(x) = x^2
        f'(x) = d/dx f(x) = 2x
        """
        ga = GeometricAlgebra(metric=dual_metric)

        batch_shape = [3, 4]

        one = ga.ones(batch_shape, kind="scalar")
        five = 5.0 * ga.ones(batch_shape, kind="scalar")
        eps = ga.ones(batch_shape, kind="pseudoscalar")

        x = one + eps
        self.assertEqual(x[""], one)
        self.assertEqual(x["0"], eps)
        self.assertAllElementsEqualTo(x.scalar, 1.0)
        self.assertEqual(x.get_part_mv("pseudoscalar"), eps)
        self.assertAllElementsEqualTo(x.get_part("scalar"), 1.0)
        self.assertAllElementsEqualTo(x.get_part("pseudoscalar"), 1.0)

        # f(1) = 1^2 = 1, f'(1) = 2
        x_squared = x * x
        self.assertAllElementsEqualTo(x_squared.scalar, 1.0)
        self.assertEqual(x_squared["0"], 2.0 * eps)

        y = five + eps
        self.assertEqual(y[""], five)
        self.assertEqual(y["0"], eps)
        self.assertAllElementsEqualTo(y.scalar, 5.0)
        self.assertEqual(y.get_part_mv("pseudoscalar"), eps)
        self.assertAllElementsEqualTo(y.get_part("scalar"), 5.0)
        self.assertAllElementsEqualTo(y.get_part("pseudoscalar"), 1.0)

        # f(5) = 5^2 = 25, f'(5) = 10
        y_squared = y * y
        self.assertAllElementsEqualTo(y_squared.scalar, 25.0)
        self.assertEqual(y_squared["0"], eps * 10.0)

    def test_mul_inverse(self):
        ga = GeometricAlgebra(metric=dual_metric)

        # a = 2
        a = ga.fill([], fill_value=2.0, kind="scalar")

        # b = 3 + 3e0
        b = ga.fill([], fill_value=3.0, kind="mv")

        # a * b = 2 * (3 + 3e0) = 6 + 6e0
        c = a * b
        self.assertEqual(c, 6.0 + 6.0 * ga.basis_mvs[0])

        # a^-1 = 1 / 2
        a_inv = a.inverse()
        self.assertEqual(a_inv, 0.5)

        # c = a * b
        # => a_inv * c = b
        self.assertEqual(a_inv * c, b)

        # Since a is scalar, should commute too.
        # => c * a_inv = b
        self.assertEqual(c * a_inv, b)

        # b is not invertible and will throw an exception
        self.assertRaises(Exception, b.inverse)
