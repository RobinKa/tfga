import unittest as ut

import tensorflow as tf

from tfga import GeometricAlgebra

# Make tensorflow not take over the entire GPU memory
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

dual_metric = [0]
dual_bases = ["0"]
dual_blades = ["", "0"]
dual_blade_degrees = [len(blade) for blade in dual_blades]


class TestDualGeometricAlgebraMultiply(ut.TestCase):
    def assertTensorsEqual(self, a, b):
        self.assertTrue(tf.reduce_all(a == b), "%s not equal to %s" % (a, b))

    def test_mul_mv_mv(self):
        ga = GeometricAlgebra(metric=dual_metric)

        zero = ga.from_scalar(0.0)
        one = ga.from_scalar(1.0)
        eps = ga.from_tensor_with_kind(tf.ones(1), kind="pseudoscalar")
        ten = ga.from_scalar(10.0)

        self.assertTensorsEqual(ga.geom_prod(eps, eps), zero)
        self.assertTensorsEqual(ga.geom_prod(one, one), one)
        self.assertTensorsEqual(ga.geom_prod(zero, one), zero)
        self.assertTensorsEqual(ga.geom_prod(one, zero), zero)
        self.assertTensorsEqual(ga.geom_prod(one, eps), eps)
        self.assertTensorsEqual(ga.geom_prod(eps, one), eps)
        self.assertTensorsEqual(ga.geom_prod(zero, zero), zero)
        self.assertTensorsEqual(ga.geom_prod(ten, zero), zero)
        self.assertTensorsEqual(ga.geom_prod(zero, ten), zero)
        self.assertTensorsEqual(ga.geom_prod(ga.geom_prod(ten, eps), eps), zero)
        self.assertTensorsEqual(ga.geom_prod(ten, one), ten)
        self.assertTensorsEqual(ga.geom_prod(one, ten), ten)

    def test_mul_tf_mv(self):
        ga = GeometricAlgebra(metric=dual_metric)

        zero = ga.from_scalar(0.0)
        one = ga.from_scalar(1.0)
        eps = ga.from_tensor_with_kind(tf.ones(1), kind="pseudoscalar")
        ten = ga.from_scalar(10.0)

        zero_tf = tf.convert_to_tensor([0, 0], dtype=tf.float32)
        one_tf = tf.convert_to_tensor([1, 0], dtype=tf.float32)
        eps_tf = tf.convert_to_tensor([0, 1], dtype=tf.float32)
        ten_tf = tf.convert_to_tensor([10, 0], dtype=tf.float32)

        self.assertTensorsEqual(ga.geom_prod(one, one_tf), one)
        self.assertTensorsEqual(ga.geom_prod(one_tf, one), one)
        self.assertTensorsEqual(ga.geom_prod(zero, one_tf), zero)
        self.assertTensorsEqual(ga.geom_prod(one_tf, zero), zero)
        self.assertTensorsEqual(ga.geom_prod(zero_tf, one), zero)
        self.assertTensorsEqual(ga.geom_prod(one, zero_tf), zero)
        self.assertTensorsEqual(ga.geom_prod(one_tf, eps), eps)
        self.assertTensorsEqual(ga.geom_prod(eps, one_tf), eps)
        self.assertTensorsEqual(ga.geom_prod(zero_tf, zero), zero)
        self.assertTensorsEqual(ga.geom_prod(zero, zero_tf), zero)
        self.assertTensorsEqual(ga.geom_prod(ten_tf, zero), zero)
        self.assertTensorsEqual(ga.geom_prod(zero, ten_tf), zero)
        self.assertTensorsEqual(ga.geom_prod(ten, zero_tf), zero)
        self.assertTensorsEqual(ga.geom_prod(zero_tf, ten), zero)
        self.assertTensorsEqual(ga.geom_prod(ga.geom_prod(ten_tf, eps), eps), zero)
        self.assertTensorsEqual(ga.geom_prod(ten_tf, one), ten)
        self.assertTensorsEqual(ga.geom_prod(one, ten_tf), ten)
        self.assertTensorsEqual(ga.geom_prod(ten, one_tf), ten)
        self.assertTensorsEqual(ga.geom_prod(one_tf, ten), ten)


class TestDualGeometricAlgebraMisc(ut.TestCase):
    def assertTensorsEqual(self, a, b):
        self.assertTrue(tf.reduce_all(a == b), "%s not equal to %s" % (a, b))

    def test_auto_diff_square(self):
        """Test automatic differentiation using
        dual numbers for the square function.
        f(x) = x^2
        f'(x) = d/dx f(x) = 2x
        """
        ga = GeometricAlgebra(metric=dual_metric)

        one = ga.from_scalar(1.0)
        five = ga.from_scalar(5.0)
        eps = ga.from_tensor_with_kind(tf.ones(1), kind="pseudoscalar")

        x = one + eps

        # f(1) = 1^2 = 1, f'(1) = 2
        x_squared = ga.geom_prod(x, x)
        self.assertTensorsEqual(ga.select_blades_with_name(x_squared, ""), 1.0)
        self.assertTensorsEqual(ga.select_blades_with_name(x_squared, "0"), 2.0)

        y = five + eps

        # f(5) = 5^2 = 25, f'(5) = 10
        y_squared = ga.geom_prod(y, y)
        self.assertTensorsEqual(ga.select_blades_with_name(y_squared, ""), 25.0)
        self.assertTensorsEqual(ga.select_blades_with_name(y_squared, "0"), 10.0)

    def test_batched_auto_diff_square(self):
        """Test automatic differentiation using
        dual numbers for the square function.
        Use batch with identical elements.
        f(x) = x^2
        f'(x) = d/dx f(x) = 2x
        """
        ga = GeometricAlgebra(metric=dual_metric)

        one = ga.from_tensor_with_kind(tf.ones([3, 4, 1]), kind="scalar")
        five = ga.from_tensor_with_kind(tf.fill([3, 4, 1], 5.0), kind="scalar")
        eps = ga.from_tensor_with_kind(tf.ones([3, 4, 1]), kind="pseudoscalar")

        x = one + eps

        # f(1) = 1^2 = 1, f'(1) = 2
        x_squared = ga.geom_prod(x, x)
        self.assertTensorsEqual(ga.select_blades_with_name(x_squared, ""), 1.0)
        self.assertTensorsEqual(ga.select_blades_with_name(x_squared, "0"), 2.0)

        y = five + eps

        # f(5) = 5^2 = 25, f'(5) = 10
        y_squared = ga.geom_prod(y, y)
        self.assertTensorsEqual(ga.select_blades_with_name(y_squared, ""), 25.0)
        self.assertTensorsEqual(ga.select_blades_with_name(y_squared, "0"), 10.0)

    def test_mul_inverse(self):
        ga = GeometricAlgebra(metric=dual_metric)

        # a = 2
        a = ga.from_tensor_with_kind(tf.fill([1], 2.0), kind="scalar")

        # b = 3 + 3e0
        b = ga.from_tensor_with_kind(tf.fill([2], 3.0), kind="mv")

        # a * b = 2 * (3 + 3e0) = 6 + 6e0
        c = ga.geom_prod(a, b)
        self.assertTensorsEqual(c, ga.from_scalar(6.0) + 6.0 * ga.e("0"))

        # a^-1 = 1 / 2
        a_inv = ga.inverse(a)
        self.assertTensorsEqual(ga.select_blades_with_name(a_inv, ""), 0.5)

        # c = a * b
        # => a_inv * c = b
        self.assertTensorsEqual(ga.geom_prod(a_inv, c), b)

        # Since a is scalar, should commute too.
        # => c * a_inv = b
        self.assertTensorsEqual(ga.geom_prod(c, a_inv), b)

        # b is not simply invertible (because it does not square to a scalar)
        # and will throw an exception
        self.assertRaises(Exception, ga.simple_inverse, b)

        # b is invertible with the shirokov inverse
        b_inv = ga.inverse(b)
        self.assertTensorsEqual(ga.geom_prod(b, b_inv), 1 * ga.e(""))
