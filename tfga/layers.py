"""Provides Geometric Algebra Keras layers."""
from typing import List, Union
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import (
    initializers, activations, regularizers, constraints
)
from .blades import BladeKind
from .tfga import GeometricAlgebra


class TensorToGeometric(layers.Layer):
    """Layer for converting tensors with given blade indices to
    geometric algebra tensors.

    Args:
        algebra: GeometricAlgebra instance to use
        blade_indices: blade indices to interpret the last axis of the
        input tensor as
    """

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
    """Layer for converting tensors with given blade kind to
    geometric algebra tensors.

    Args:
        algebra: GeometricAlgebra instance to use
        kind: blade kind indices to interpret the last axis of the
        input tensor as
    """

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
    """Layer for extracting given blades from geometric algebra tensors.

    Args:
        algebra: GeometricAlgebra instance to use
        blade_indices: blade indices to extract
    """

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
    """Layer for extracting blades of a kind from geometric algebra tensors.

    Args:
        algebra: GeometricAlgebra instance to use
        kind: blade indices of kind to extract
    """

    def __init__(self, algebra: GeometricAlgebra, kind: BladeKind,
                 **kwargs):
        blade_indices = algebra.get_kind_blade_indices(kind)
        super().__init__(algebra=algebra, blade_indices=blade_indices,
                         **kwargs)


class GeometricProductDense(layers.Layer):
    """Analagous to Keras' Dense layer but using multivector-valued matrices
    instead of scalar ones and geometric multiplication instead of standard
    multiplication.

    Args:
        algebra: GeometricAlgebra instance to use for the parameters
        blade_indices_kernel: Blade indices to use for the kernel parameter
        blade_indices_bias: Blade indices to use for the bias parameter (if used)
    """

    def __init__(
        self,
        algebra: GeometricAlgebra,
        units: int,
        blade_indices_kernel: List[int],
        blade_indices_bias: Union[None, List[int]] = None,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        self.algebra = algebra
        self.units = units
        self.blade_indices_kernel = tf.convert_to_tensor(
            blade_indices_kernel, dtype_hint=tf.int64)
        if use_bias:
            self.blade_indices_bias = tf.convert_to_tensor(
                blade_indices_bias, dtype_hint=tf.int64)

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape: tf.TensorShape):
        self.num_input_units = input_shape[-2]
        shape_kernel = [
            self.units,
            self.num_input_units,
            self.blade_indices_kernel.shape[0]
        ]
        self.kernel = self.add_weight(
            "kernel",
            shape=shape_kernel,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True
        )
        if self.use_bias:
            shape_bias = [self.units, self.blade_indices_bias.shape[0]]
            self.bias = self.add_weight(
                "bias",
                shape=shape_bias,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True
            )
        else:
            self.bias = None
        self.built = True

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([*input_shape[:-2], self.units, self.algebra.num_blades])

    def call(self, inputs):
        w_geom = self.algebra.from_tensor(
            self.kernel, self.blade_indices_kernel)

        # Perform a matrix-multiply, but using geometric product instead of
        # standard multiplication. To do this we do the geometric product
        # elementwise and then sum over the common axis.
        # [..., 1, I, X] * [..., O, I, X] -> [..., O, I, X] -> [..., O, X]
        inputs_expanded = tf.expand_dims(inputs, axis=inputs.shape.ndims - 2)
        result = tf.reduce_sum(self.algebra.geom_prod(
            inputs_expanded, w_geom), axis=-2)

        if self.bias is not None:
            b_geom = self.algebra.from_tensor(
                self.bias, self.blade_indices_bias)
            result += b_geom

        return self.activation(result)

    def get_config(self):
        layers.Dense
        config = super().get_config()
        config.update({
            "algebra":
                self.algebra,  # TODO: Probably isn't right.
            "blade_indices_kernel":
                self.blade_indices_kernel,
            "blade_indices_bias":
                self.blade_indices_bias,
            "units":
                self.units,
            "activation":
                activations.serialize(self.activation),
            "use_bias":
                self.use_bias,
            "kernel_initializer":
                initializers.serialize(self.kernel_initializer),
            "bias_initializer":
                initializers.serialize(self.bias_initializer),
            "kernel_regularizer":
                regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer":
                regularizers.serialize(self.bias_regularizer),
            "activity_regularizer":
                regularizers.serialize(self.activity_regularizer),
            "kernel_constraint":
                constraints.serialize(self.kernel_constraint),
            "bias_constraint":
                constraints.serialize(self.bias_constraint)
        })
        return config


class GeometricSandwichProductDense(GeometricProductDense):
    """Analagous to Keras' Dense layer but using multivector-valued matrices
    instead of scalar ones and geometric sandwich multiplication instead of
    standard multiplication.

    Args:
        algebra: GeometricAlgebra instance to use for the parameters
        blade_indices_kernel: Blade indices to use for the kernel parameter
        blade_indices_bias: Blade indices to use for the bias parameter (if used)
    """

    def __init__(
        self, algebra, units, blade_indices_kernel, blade_indices_bias=None,
        activation=None, use_bias=True, kernel_initializer="glorot_uniform",
        bias_initializer="zeros", kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, **kwargs
    ):
        super().__init__(
            algebra, units,
            blade_indices_kernel,
            blade_indices_bias=blade_indices_bias,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, **kwargs
        )

    def call(self, inputs):
        w_geom = self.algebra.from_tensor(
            self.kernel, self.blade_indices_kernel)

        # Same as GeometricProductDense but using R*x*~R instead of just R*x
        inputs_expanded = tf.expand_dims(inputs, axis=inputs.shape.ndims - 2)
        result = tf.reduce_sum(
            self.algebra.geom_prod(
                w_geom,
                self.algebra.geom_prod(
                    inputs_expanded,
                    self.algebra.reversion(w_geom)
                )
            ),
            axis=-2
        )

        if self.bias is not None:
            b_geom = self.algebra.from_tensor(
                self.bias, self.blade_indices_bias)
            result += b_geom

        return self.activation(result)
