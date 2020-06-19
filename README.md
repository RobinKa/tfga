# TFGA - TensorFlow Geometric Algebra
[![Build status](https://github.com/RobinKa/tfga/workflows/Build%20Test%20Publish/badge.svg)](https://github.com/RobinKa/tfga/actions) [![PyPI](https://badge.fury.io/py/tfga.svg)](https://badge.fury.io/py/tfga)

[GitHub](https://github.com/RobinKa/tfga) | [Docs](https://tfga.warlock.ai) | [Benchmarks](https://github.com/RobinKa/tfga/tree/master/benchmarks) | [Slides](https://tfgap.warlock.ai)

Python package for Geometric / Clifford Algebra with TensorFlow 2.

**This project is a work in progress. Its API may change and the examples aren't polished yet.**

Pull requests and suggestions either by opening an issue or by [sending me an email](mailto:tora@warlock.ai) are welcome.

## Installation
Install using pip: `pip install tfga`

Requirements:
- Python 3
- tensorflow 2
- numpy

## Basic usage
There are two ways to use this library. In both ways we first create a `GeometricAlgebra` instance given a metric.
Then we can either work on `tf.Tensor` instances directly where the last axis is assumed to correspond to
the algebra's blades.
```python
import tensorflow as tf
from tfga import GeometricAlgebra

# Create an algebra with 3 basis vectors given their metric.
# Contains geometric algebra operations.
ga = GeometricAlgebra(metric=[1, 1, 1])

# Create geometric algebra tf.Tensor for vector blades (ie. e_0 + e_1 + e_2).
# Represented as tf.Tensor with shape [8] (one value for each blade of the algebra).
# tf.Tensor: [0, 1, 1, 1, 0, 0, 0, 0]
ordinary_vector = ga.from_tensor_with_kind(tf.ones(3), kind="vector")

# 5 + 5 e_01 + 5 e_02 + 5 e_12
quaternion = ga.from_tensor_with_kind(tf.fill(dims=4, value=5), kind="even")

# 5 + 1 e_0 + 1 e_1 + 1 e_2 + 5 e_01 + 5 e_02 + 5 e_12
multivector = ordinary_vector + quaternion

# Inner product e_0 | (e_0 + e_1 + e_2) = 1
# ga.print is like print, but has extra formatting for geometric algebra tf.Tensor instances.
ga.print(ga.inner_prod(ga.e0, ordinary_vector))

# Exterior product e_0 ^ e_1 = e_01.
ga.print(ga.ext_prod(ga.e0, ga.e1))

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

Alternatively we can convert the geometric algebra `tf.Tensor` instance to `MultiVector`
instances which wrap the operations and provide operator overrides for convenience.
This can be done by using the `__call__` operator of the `GeometricAlgebra` instance.
```python
# Create geometric algebra tf.Tensor instances
a = ga.e123
b = ga.e1

# Wrap them as `MultiVector` instances
mv_a = ga(a)
mv_b = ga(b)

# Reversion ((~mv_a).tensor equivalent to ga.reversion(a))
print(~mv_a)

# Geometric / inner / exterior product
print(mv_a * mv_b)
print(mv_a | mv_b)
print(mv_a ^ mv_b)
```

## Keras layers
TFGA also provides [Keras](https://www.tensorflow.org/guide/keras/overview) layers which provide
layers similar to the existing ones but using multivectors instead. For example the GeometricProductDense
layer is exactly the same as the [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) layer but uses
multivector-valued weights and biases instead of scalar ones. The exact kind of multivector-type can be
passed too. Example:

```python
import tensorflow as tf
from tfga import GeometricAlgebra
from tfga.layers import TensorToGeometric, GeometricToTensor, GeometricProductDense

# 4 basis vectors (e0^2=+1, e1^2=-1, e2^2=-1, e3^2=-1)
sta = GeometricAlgebra([1, -1, -1, -1])

# We want our dense layer to perform a matrix multiply
# with a matrix that has vector-valued entries.
vector_blade_indices = sta.get_kind_blade_indices(BladeKind.VECTOR),

# Create our input of shape [Batch, Units, BladeValues]
tensor = tf.ones([20, 6, 4])

# The matrix-multiply will perform vector * vector
# so our result will be scalar + bivector.
# Use the resulting blade type for the bias too which is
# added to the result.
result_indices = tf.concat([
    sta.get_kind_blade_indices(BladeKind.SCALAR), # 1 index
    sta.get_kind_blade_indices(BladeKind.BIVECTOR) # 6 indices
], axis=0)

sequence = tf.keras.Sequential([
    # Converts the last axis to a dense multivector
    # (so, 4 -> 16 (total number of blades in the algebra))
    TensorToGeometric(sta, blade_indices=vector_blade_indices),
    # Perform matrix multiply with vector-valued matrix
    GeometricProductDense(
        algebra=sta, units=8, # units is analagous to Keras' Dense layer
        blade_indices_kernel=vector_blade_indices,
        blade_indices_bias=result_indices
    ),
    # Extract our wanted blade indices (last axis 16 -> 7 (1+6))
    GeometricToTensor(sta, blade_indices=result_indices)
])

# Result will have shape [20, 8, 7]
result = sequence(tensor)
```

For performing a geometric sandwich product `R * x * ~R` instead of just the geometric product `R * x`
there also exists the `GeometricSandwichProductDense` with an identical API.

## Notebooks
[Generic examples](https://github.com/RobinKa/tfga/tree/master/notebooks/tfga.ipynb)

[Using Keras layers to estimate triangle area](https://github.com/RobinKa/tfga/tree/master/notebooks/keras-triangles.ipynb)

[Classical Electromagnetism using Geometric Algebra](https://github.com/RobinKa/tfga/tree/master/notebooks/em.ipynb)

[Quantum Electrodynamics using Geometric Algebra](https://github.com/RobinKa/tfga/tree/master/notebooks/qed.ipynb)

[Projective Geometric Algebra](https://github.com/RobinKa/tfga/tree/master/notebooks/pga.ipynb)

## Tests
Tests using Python's built-in `unittest` module are available in the `tests` directory. All tests can be run by
executing `python -m unittest discover tests` from the root directory of the repository.