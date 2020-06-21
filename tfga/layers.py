"""Provides Geometric Algebra Keras layers."""
from typing import List, Union
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import (
    initializers, activations, regularizers, constraints
)
from tensorflow.keras.utils import register_keras_serializable
from .blades import BladeKind
from .tfga import GeometricAlgebra


class GeometricAlgebraLayer(layers.Layer):
    def __init__(self, algebra: GeometricAlgebra, **kwargs):
        self.algebra = algebra
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        # Create algebra if necessary (should only occur once, assumes that
        # config is actually mutable).
        if "algebra" not in config:
            assert "metric" in config
            config["algebra"] = GeometricAlgebra(config["metric"])
            del config["metric"]
        return cls(**config)

    def get_config(self):
        # Store metric of the algebra. In from_config() we will recreate the
        # algebra from the metric.
        config = super().get_config()
        config.update({
            "metric": self.algebra.metric.numpy()
        })
        return config


@register_keras_serializable(package="TFGA")
class TensorToGeometric(GeometricAlgebraLayer):
    """Layer for converting tensors with given blade indices to
    geometric algebra tensors.

    Args:
        algebra: GeometricAlgebra instance to use
        blade_indices: blade indices to interpret the last axis of the
        input tensor as
    """

    def __init__(self, algebra: GeometricAlgebra, blade_indices: List[int],
                 **kwargs):
        super().__init__(algebra=algebra, **kwargs)
        self.blade_indices = tf.convert_to_tensor(
            blade_indices, dtype=tf.int64)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([*input_shape[:-1], self.algebra.num_blades])

    def call(self, inputs):
        return self.algebra.from_tensor(inputs, blade_indices=self.blade_indices)

    def get_config(self):
        config = super().get_config()
        config.update({
            "blade_indices": self.blade_indices.numpy()
        })
        return config


@register_keras_serializable(package="TFGA")
class TensorWithKindToGeometric(GeometricAlgebraLayer):
    """Layer for converting tensors with given blade kind to
    geometric algebra tensors.

    Args:
        algebra: GeometricAlgebra instance to use
        kind: blade kind indices to interpret the last axis of the
        input tensor as
    """

    def __init__(self, algebra: GeometricAlgebra, kind: BladeKind,
                 **kwargs):
        super().__init__(algebra=algebra, **kwargs)
        self.kind = kind

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([*input_shape[:-1], self.algebra.get_kind_blade_indices(self.kind).shape[0]])

    def call(self, inputs):
        return self.algebra.from_tensor_with_kind(inputs, kind=self.kind)

    def get_config(self):
        config = super().get_config()
        config.update({
            "kind": self.kind
        })
        return config


@register_keras_serializable(package="TFGA")
class GeometricToTensor(GeometricAlgebraLayer):
    """Layer for extracting given blades from geometric algebra tensors.

    Args:
        algebra: GeometricAlgebra instance to use
        blade_indices: blade indices to extract
    """

    def __init__(self, algebra: GeometricAlgebra, blade_indices: List[int],
                 **kwargs):
        super().__init__(algebra=algebra, **kwargs)
        self.blade_indices = tf.convert_to_tensor(
            blade_indices, dtype=tf.int64)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([*input_shape[:-1], self.blade_indices.shape[0]])

    def call(self, inputs):
        return tf.gather(inputs, self.blade_indices, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "blade_indices": self.blade_indices.numpy()
        })
        return config


@register_keras_serializable(package="TFGA")
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


@register_keras_serializable(package="TFGA")
class GeometricProductDense(GeometricAlgebraLayer):
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
        super().__init__(algebra=algebra, activity_regularizer=activity_regularizer, **kwargs)

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
        config = super().get_config()
        config.update({
            "blade_indices_kernel":
                self.blade_indices_kernel.numpy(),
            "blade_indices_bias":
                self.blade_indices_bias.numpy(),
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


@register_keras_serializable(package="TFGA")
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


@register_keras_serializable(package="TFGA")
class GeometricProductElementwise(GeometricAlgebraLayer):
    """Performs the elementwise geometric product with a list of multivectors
    with as many elements as there are input units.

    Args:
        algebra: GeometricAlgebra instance to use for the parameters
        blade_indices_kernel: Blade indices to use for the kernel parameter
        blade_indices_bias: Blade indices to use for the bias parameter (if used)
    """

    def __init__(
        self,
        algebra: GeometricAlgebra,
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
        super().__init__(algebra=algebra, activity_regularizer=activity_regularizer, **kwargs)

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
            shape_bias = [self.num_input_units,
                          self.blade_indices_bias.shape[0]]
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
        return tf.TensorShape([*input_shape[:-1], self.algebra.num_blades])

    def call(self, inputs):
        w_geom = self.algebra.from_tensor(
            self.kernel, self.blade_indices_kernel)

        # Elementwise multiplication for each unit with a multivector.
        # [..., U, X] * [U, X] -> [..., U, X]
        result = self.algebra.geom_prod(inputs, w_geom)

        if self.bias is not None:
            b_geom = self.algebra.from_tensor(
                self.bias, self.blade_indices_bias)
            result += b_geom

        return self.activation(result)

    def get_config(self):
        config = super().get_config()
        config.update({
            "blade_indices_kernel":
                self.blade_indices_kernel.numpy(),
            "blade_indices_bias":
                self.blade_indices_bias.numpy(),
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


@register_keras_serializable(package="TFGA")
class GeometricSandwichProductElementwise(GeometricProductElementwise):
    """Performs the elementwise geometric sandwich product with a list of
    multivectors with as many elements as there are input units.

    Args:
        algebra: GeometricAlgebra instance to use for the parameters
        blade_indices_kernel: Blade indices to use for the kernel parameter
        blade_indices_bias: Blade indices to use for the bias parameter (if used)
    """

    def __init__(
        self, algebra, blade_indices_kernel, blade_indices_bias=None,
        activation=None, use_bias=True, kernel_initializer="glorot_uniform",
        bias_initializer="zeros", kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None, **kwargs
    ):
        super().__init__(
            algebra,
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

        # Elementwise multiplication Rx~R for each unit with a multivector.
        # [..., U, X] * [U, X] -> [..., U, X]
        result = self.algebra.geom_prod(
            w_geom,
            self.algebra.geom_prod(
                inputs,
                self.algebra.reversion(w_geom)
            )
        )

        if self.bias is not None:
            b_geom = self.algebra.from_tensor(
                self.bias, self.blade_indices_bias)
            result += b_geom

        return self.activation(result)


@register_keras_serializable(package="TFGA")
class GeometricProductConv1D(GeometricAlgebraLayer):
    """Analagous to Keras' Conv1D layer but using multivector-valued kernels
    instead of scalar ones and geometric product instead of
    standard multiplication.

    Args:
        algebra: GeometricAlgebra instance to use for the parameters
        filters: How many channels the output will have
        kernel_size: Size for the convolution kernel
        stride: Stride to use for the convolution
        padding: "SAME" (zero-pad input length so output
            length == input length / stride) or "VALID" (no padding)
        blade_indices_kernel: Blade indices to use for the kernel parameter
        blade_indices_bias: Blade indices to use for the bias parameter (if used)
    """

    def __init__(
        self,
        algebra: GeometricAlgebra,
        filters: int,
        kernel_size: int,
        stride: int,
        padding: str,
        blade_indices_kernel: List[int],
        blade_indices_bias: Union[None, List[int]] = None,
        dilations: Union[None, int] = None,
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
        super().__init__(
            algebra=algebra,
            activity_regularizer=activity_regularizer,
            **kwargs
        )

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilations = dilations

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
        # I: [..., S, C, B]
        self.num_input_filters = input_shape[-2]

        # K: [K, IC, OC, B]
        shape_kernel = [
            self.kernel_size,
            self.num_input_filters,
            self.filters,
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
            shape_bias = [self.filters, self.blade_indices_bias.shape[0]]
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

    def call(self, inputs):
        k_geom = self.algebra.from_tensor(
            self.kernel, self.blade_indices_kernel)

        result = self.algebra.geom_conv1d(
            inputs, k_geom,
            stride=self.stride, padding=self.padding,
            dilations=self.dilations
        )

        if self.bias is not None:
            b_geom = self.algebra.from_tensor(
                self.bias, self.blade_indices_bias)
            result += b_geom

        return self.activation(result)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters":
                self.filters,
            "kernel_size":
                self.kernel_size,
            "stride":
                self.stride,
            "padding":
                self.padding,
            "dilations":
                self.dilations,
            "blade_indices_kernel":
                self.blade_indices_kernel.numpy(),
            "blade_indices_bias":
                self.blade_indices_bias.numpy(),
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


@register_keras_serializable(package="TFGA")
class GeometricAlgebraExp(GeometricAlgebraLayer):
    """Calculates the exponential function of the input. Input must square to
    a scalar.

    Args:
        algebra: GeometricAlgebra instance to use
        square_scalar_tolerance: Tolerance to use for the square scalar check
            or None if the check should be skipped
    """

    def __init__(
        self,
        algebra: GeometricAlgebra,
        square_scalar_tolerance: Union[float, None] = 1e-4,
        **kwargs
    ):
        super().__init__(algebra=algebra, **kwargs)
        self.square_scalar_tolerance = square_scalar_tolerance

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([*input_shape[:-1], self.algebra.num_blades])

    def call(self, inputs):
        return self.algebra.exp(
            inputs, square_scalar_tolerance=self.square_scalar_tolerance
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "square_scalar_tolerance": self.square_scalar_tolerance
        })
        return config
