from clifford.sta import D, D_blades
from clifford import MVArray
import numpy as np
import pytest
import numba


def _clifford_add(a, b):
    return a + b


@numba.njit(parallel=True, nogil=True)
def _clifford_raw_add(a, b):
    return a + b


def _clifford_mul(a, b):
    return a * b


gmt_func = D.gmt_func
@numba.njit(parallel=True, nogil=True)
def _clifford_raw_mul(a, b):
    op = np.zeros(a.shape)
    for i in numba.prange(op.shape[0]):
        op[i, :] = gmt_func(a[i, :], b[i, :])
    return a * b


def _mv_ones(num_elements):
    return MVArray([D.MultiVector(value=np.ones(2**4, dtype=np.float32)) for i in range(num_elements)])


@pytest.mark.parametrize("num_elements", [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
def test_clifford_add_mv_mv(num_elements, benchmark):
    a = _mv_ones(num_elements)
    b = _mv_ones(num_elements)
    benchmark(_clifford_add, a, b)


@pytest.mark.parametrize("num_elements", [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
def test_clifford_raw_mul_mv_mv(num_elements, benchmark):
    a = _mv_ones(num_elements).value
    b = _mv_ones(num_elements).value
    benchmark(_clifford_raw_mul, a, b)


@pytest.mark.parametrize("num_elements", [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
def test_clifford_raw_add_mv_mv(num_elements, benchmark):
    a = _mv_ones(num_elements).value
    b = _mv_ones(num_elements).value
    benchmark(_clifford_raw_add, a, b)


@pytest.mark.parametrize("num_elements", [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000])
def test_clifford_mul_mv_mv(num_elements, benchmark):
    a = _mv_ones(num_elements)
    b = _mv_ones(num_elements)
    benchmark(_clifford_mul, a, b)


if __name__ == '__main__':
    pytest.main()