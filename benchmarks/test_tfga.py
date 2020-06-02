from tfga import GeometricAlgebra
import tensorflow as tf
import pytest


def _tfga_add(a, b):
    return a + b


def _tfga_mul(ga, a, b):
    return ga.geom_prod(a, b)


@pytest.mark.parametrize("num_elements", [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
def test_tfga_add_mv_mv(num_elements, benchmark):
    ga = GeometricAlgebra([1, -1, -1, -1])
    a = tf.ones([num_elements, ga.num_blades])
    b = tf.ones([num_elements, ga.num_blades])
    benchmark(_tfga_add, a, b)


@pytest.mark.parametrize("num_elements", [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
def test_tfga_mul_mv_mv(num_elements, benchmark):
    ga = GeometricAlgebra([1, -1, -1, -1])
    a = tf.ones([num_elements, ga.num_blades])
    b = tf.ones([num_elements, ga.num_blades])
    benchmark(_tfga_mul, ga, a, b)
