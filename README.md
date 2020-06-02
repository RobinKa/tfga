# TFGA - TensorFlow Geometric Algebra
[![Build status](https://github.com/RobinKa/tfga/workflows/Build%20Test%20Publish/badge.svg)](https://github.com/RobinKa/tfga/actions) [![PyPI](https://badge.fury.io/py/tfga.svg)](https://badge.fury.io/py/tfga)

[GitHub](https://github.com/RobinKa/tfga) | [Docs](https://tfga.warlock.ai) | [Benchmarks](https://github.com/RobinKa/tfga/tree/master/benchmarks)

Python package for Geometric / Clifford Algebra with TensorFlow 2.

This project is a work in progress. Its API may change and the examples aren't polished yet.
Pull requests and suggestions either by opening an issue or by [sending me an email](mailto:tora@warlock.ai) are welcome.

## Installation
Install using pip: `pip install tfga`

Requirements:
- Python 3
- tensorflow 2
- numpy

## Basic usage
```python
import tensorflow as tf
from tfga import GeometricAlgebra

# Create an algebra with 3 basis vectors given their metric.
# Contains geometric algebra operations.
ga = GeometricAlgebra(metric=[1, 1, 1])

# 1 e_0 + 1 e_1 + 1 e_2
ordinary_vector = ga.from_tensor_with_kind(tf.ones(3), kind="vector")

# 5 + 5 e_01 + 5 e_02 + 5 e_12
quaternion = ga.from_tensor_with_kind(tf.fill(dims=4, value=5), kind="even")

# 5 + 1 e_0 + 1 e_1 + 1 e_2 + 5 e_01 + 5 e_02 + 5 e_12
multivector = ordinary_vector + quaternion

# Inner product e_0 | 1 e_0 + 1 e_1 + 1 e_2 = 1
# ga.print is like print, but has extra formatting for tf.Tensors
# that are shaped like multivectors (ie. here has 0 elements on last axis).
ga.print(ga.inner_prod(ga.basis_mvs[0], ordinary_vector))

# Exterior product e_0 ^ e_1 = e_01
ga.print(ga.ext_prod(ga.basis_mvs[0], ga.basis_mvs[1]))

# Grade reversal ~(5 + 5 e_01 + 5 e_02 + 5 e_12)
# = 5 + 5 e_10 + 5 e_20 + 5 e_21
# = 5 - 5 e_01 - 5 e_02 - 5 e_12
ga.print(ga.reversion(quaternion))

# tf.Tensor 5
ga.print(quaternion[0])

# tf.Tensor of shape [1]: -5 (ie. reversed sign of e_01 component)
ga.print(ga.select_blades(quaternion, "10"))

# tf.Tensor of shape [8] with only e_01 component equal to 5
ga.print(ga.keep_blades(quaternion, "10"))
```

## Notebooks
[Generic examples](https://github.com/RobinKa/tfga/tree/master/notebooks/tfga.ipynb)

[Quantum Electrodynamics using Geometric Algebra](https://github.com/RobinKa/tfga/tree/master/notebooks/qed.ipynb)

[Projective Geometric Algebra](https://github.com/RobinKa/tfga/tree/master/notebooks/pga.ipynb)