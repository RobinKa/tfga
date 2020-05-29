from tfga import GeometricAlgebra
import pytest

def _tfga_add(a, b):
    return a + b


def _tfga_mul(a, b):
    return a * b


@pytest.mark.parametrize("num_elements", [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
def test_tfga_add_mv_mv(num_elements, benchmark):
    ga = GeometricAlgebra([-1, 1, 1, 1])
    a = ga.ones([num_elements])
    b = ga.ones([num_elements])
    benchmark(_tfga_add, a, b)


@pytest.mark.parametrize("num_elements", [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
def test_tfga_mul_mv_mv(num_elements, benchmark):
    ga = GeometricAlgebra([-1, 1, 1, 1])
    a = ga.ones([num_elements])
    b = ga.ones([num_elements])
    benchmark(_tfga_mul, a, b)
