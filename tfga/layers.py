"""Provides Geometric Algebra Keras layers."""
from typing import List
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from .blades import BladeKind
from .tfga import GeometricAlgebra


class GeometricTensorInitializer(initializers.Initializer):
    def __init__(self, algebra, blade_indices, base_initializer: initializers.Initializer):
        self._base_initializer = base_initializer
        self._algebra = algebra
        self._blade_indices = tf.convert_to_tensor(
            blade_indices, dtype_hint=tf.int64)

    def __call__(self, shape, dtype=None):
        assert shape[-1] == self._algebra.num_blades

        values = self._base_initializer(
            shape=[*shape[:-1], self._blade_indices.shape[0]], dtype=dtype
        )

        return self._algebra.from_tensor(values, blade_indices=self._blade_indices)

    def get_config(self):
        config = super().get_config()
        config.update({
            "blade_indices": self._blade_indices,
            "base_initializer": self._base_initializer
        })
        return config


class TensorToGeometric(layers.Layer):
    def __init__(self, algebra: GeometricAlgebra, blade_indices: List[int],
                 **kwargs):
        super().__init__(**kwargs)

        self._algebra = algebra
        self._blade_indices = blade_indices

    def call(self, inputs):
        return self._algebra.from_tensor(inputs, blade_indices=self._blade_indices)

    def get_config(self):
        config = super().get_config()
        config.update({
            "blade_indices": self._blade_indices
        })
        return config


class TensorWithKindToGeometric(layers.Layer):
    def __init__(self, algebra: GeometricAlgebra, kind: BladeKind,
                 **kwargs):
        super().__init__(**kwargs)

        self._algebra = algebra
        self._kind = kind

    def call(self, inputs):
        return self._algebra.from_tensor_with_kind(inputs, kind=self._kind)

    def get_config(self):
        config = super().get_config()
        config.update({
            "kind": self._kind
        })
        return config


class GeometricToTensor(layers.Layer):
    def __init__(self, algebra: GeometricAlgebra, blade_indices: List[int],
                 **kwargs):
        super().__init__(**kwargs)

        self._algebra = algebra
        self._blade_indices = blade_indices

    def call(self, inputs):
        return tf.gather(inputs, self._blade_indices, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "blade_indices": self._blade_indices
        })
        return config


class GeometricToTensorWithKind(GeometricToTensor):
    def __init__(self, algebra: GeometricAlgebra, kind: BladeKind,
                 **kwargs):
        blade_indices = algebra.get_kind_blade_indices(kind)
        super().__init__(algebra=algebra, blade_indices=blade_indices,
                         **kwargs)


class GeometricProductDense(layers.Layer):
    def __init__(self, algebra: GeometricAlgebra, units: int,
                 blade_indices_w: List[int], blade_indices_b: List[int],
                 **kwargs):
        super().__init__(**kwargs)

        self._algebra = algebra
        self._units = units
        self._blade_indices_w = blade_indices_w
        self._blade_indices_b = blade_indices_b

    def build(self, input_shape: tf.TensorShape):
        self._num_input_units = input_shape[-2]
        initializer_w = GeometricTensorInitializer(
            self._algebra, self._blade_indices_w, initializers.RandomNormal())
        initializer_b = GeometricTensorInitializer(
            self._algebra, self._blade_indices_b, initializers.RandomNormal())
        shape_w = [self._units, self._num_input_units,
                   self._algebra.num_blades]
        shape_b = [self._units, self._algebra.num_blades]
        self._w = self.add_weight(shape=shape_w, initializer=initializer_w)
        self._b = self.add_weight(shape=shape_b, initializer=initializer_b)

    def call(self, inputs):
        # [..., 1, I, X] * [..., O, I, X] -> [..., O, I, X] -> [..., O, X]
        return tf.reduce_sum(self._algebra.geom_prod(tf.expand_dims(inputs, axis=inputs.shape.ndims - 2), self._w), axis=-2) + self._b

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self._units,
            "kind_w": self._kind_w,
            "kind_b": self._kind_b,
            "algebra": self._algebra
        })
        return config
