# TFGA - TensorFlow Geometric Algebra
Python package for Geometric / Clifford Algebra with TensorFlow 2.

[![Build status](https://github.com/RobinKa/tfga/workflows/pypi-publish/badge.svg)](https://github.com/RobinKa/tfga/actions) [![PyPI](https://badge.fury.io/py/tfga.svg)](https://badge.fury.io/py/tfga)

[GitHub](https://github.com/RobinKa/tfga) **Docs (coming soon)**

## Installation
Install using pip: `pip install tfga`

Requirements:
- Python 3
- tensorflow 2
- numpy
- scipy (optional, for `approx_pow`)

## Basic usage
```python
from tfga import GeometricAlgebra

ga = GeometricAlgebra(metric=[1, 1, 1])

# 1 e_0 + 1 e_1 + 1 e_2
ordinary_vector = ga.ones(batch_shape=[], kind="vector")

# 5 + 5 e_01 + 5 e_02 + 5.0 e_12
quaternion = ga.fill(batch_shape=[], fill_value=5.0, kind="even")

# 5 + 1 e_0 + 1 e_1 + 1 e_2 + 5 e_01 + 5 e_02 + 5.0 e_12
multivector = ordinary_vector + quaternion

# Inner product e_0 | 1 e_0 + 1 e_1 + 1 e_2 = 1
print(ga.basis_mvs[0] | ordinary_vector)

# Exterior product e_0 ^ e_1 = e_01
print(ga.basis_mvs[0] ^ ga.basis_mvs[1])

# Grade reversal ~(5 + 5 e_01 + 5 e_02 + 5.0 e_12)
# = 5 + 5 e_10 + 5 e_20 + 5.0 e_21
# = 5 - 5 e_01 - 5 e_02 - 5.0 e_12
print(~quaternion)
```

## Notebooks
[Generic examples](notebooks/tfga.ipynb)

[Quantum Electrodynamics using Geometric Algebra](notebooks/qed.ipynb)