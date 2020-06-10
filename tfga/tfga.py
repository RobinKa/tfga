"""Provides classes and operations for performing geometric algebra
with TensorFlow.

The `GeometricAlgebra` class is used to construct the algebra given a metric.
It exposes methods for operating on `tf.Tensor` instances where their last
axis is interpreted as blades of the algebra.
"""
from typing import List, Any, Union
import numbers
import tensorflow as tf
import numpy as np

from .cayley import get_cayley_tensor, blades_from_bases
from .blades import (
    BladeKind, get_blade_of_kind_indices, get_blade_indices_from_names,
    get_blade_repr, invert_blade_indices
)
from .mv_ops import mv_multiply, mv_reversion, mv_grade_automorphism
from .mv import MultiVector


class GeometricAlgebra:
    """Class used for performing geometric algebra operations on `tf.Tensor` instances.
    Exposes methods for operating on `tf.Tensor` instances where their last
    axis is interpreted as blades of the algebra.
    Holds the metric and other quantities derived from it.
    """

    def __init__(self, metric: List[float]):
        """Creates a GeometricAlgebra object given a metric.
        The algebra will have as many basis vectors as there are
        elements in the metric.

        Args:
            metric: Metric as a list. Specifies what basis vectors square to
        """
        self._metric = tf.convert_to_tensor(metric, dtype=tf.float32)

        self._num_bases = len(metric)
        self._bases = list(map(str, range(self._num_bases)))

        self._blades, self._blade_degrees = blades_from_bases(self._bases)
        self._blade_degrees = tf.convert_to_tensor(self._blade_degrees)
        self._num_blades = len(self._blades)
        self._max_degree = tf.reduce_max(self._blade_degrees)

        # [Blades, Blades, Blades]
        self._cayley, self._cayley_inner, self._cayley_outer = tf.convert_to_tensor(
            get_cayley_tensor(self.metric, self._bases, self._blades),
            dtype=tf.float32
        )

        self._blade_mvs = tf.eye(self._num_blades)
        self._basis_mvs = self._blade_mvs[1:1+self._num_bases]

        # Find the dual by looking at the anti-diagonal in the Cayley tensor.
        self._dual_blade_indices = []
        self._dual_blade_signs = []

        for blade_index in range(self._num_blades):
            dual_index = self.num_blades - blade_index - 1
            anti_diag = self._cayley[blade_index, dual_index]
            dual_sign = tf.gather(anti_diag, tf.where(
                anti_diag != 0.0)[..., 0])[..., 0]
            self._dual_blade_indices.append(dual_index)
            self._dual_blade_signs.append(dual_sign)

        self._dual_blade_indices = tf.convert_to_tensor(
            self._dual_blade_indices, dtype=tf.int64)
        self._dual_blade_signs = tf.convert_to_tensor(
            self._dual_blade_signs, dtype=tf.float32)

    def print(self, *args, **kwargs):
        """Same as the default `print` function but formats `tf.Tensor`
        instances that have as many elements on their last axis
        as the algebra has blades using `mv_repr()`.
        """
        def _is_mv(arg):
            return isinstance(arg, tf.Tensor) and arg.shape.ndims > 0 and arg.shape[-1] == self.num_blades
        new_args = [self.mv_repr(arg) if _is_mv(arg) else arg for arg in args]

        print(*new_args, **kwargs)

    @property
    def metric(self) -> tf.Tensor:
        """Metric list which contains the number that each
        basis vector in the algebra squares to
        (ie. the diagonal of the metric tensor).
        """
        return self._metric

    @property
    def cayley(self) -> tf.Tensor:
        """`MxMxM` tensor where `M` is the number of basis
        vectors in the algebra. Used for calculating the
        geometric product:

        `a_i, b_j, cayley_ijk -> c_k`
        """
        return self._cayley

    @property
    def cayley_inner(self) -> tf.Tensor:
        """Analagous to cayley but for inner product."""
        return self._cayley_inner

    @property
    def cayley_outer(self) -> tf.Tensor:
        """Analagous to cayley but for outer product."""
        return self._cayley_outer

    @property
    def blades(self) -> List[str]:
        """List of all blade names.

        Blades are all possible independent combinations of
        basis vectors. Basis vectors are named starting
        from `"0"` and counting up. The scalar blade is the
        empty string `""`.

        Example
        - Bases: `["0", "1", "2"]`
        - Blades: `["", "0", "1", "2", "01", "02", "12", "012"]`
        """
        return self._blades

    @property
    def blade_mvs(self) -> tf.Tensor:
        """List of all blade tensors in the algebra."""
        return self._blade_mvs

    @property
    def dual_blade_indices(self) -> tf.Tensor:
        """Indices of the dual blades for each blade."""
        return self._dual_blade_indices

    @property
    def dual_blade_signs(self) -> tf.Tensor:
        """Signs of the dual blades for each blade."""
        return self._dual_blade_signs

    @property
    def num_blades(self) -> int:
        """Total number of blades in the algebra."""
        return self._num_blades

    @property
    def blade_degrees(self) -> tf.Tensor:
        """List of blade-degree for each blade in the algebra."""
        return self._blade_degrees

    @property
    def max_degree(self) -> int:
        """Highest blade degree in the algebra."""
        return self._max_degree

    @property
    def basis_mvs(self) -> tf.Tensor:
        """List of basis vectors as tf.Tensor."""
        return self._basis_mvs

    def get_kind_blade_indices(self, kind: BladeKind, invert: bool = False) -> tf.Tensor:
        """Find all indices of blades of a given kind in the algebra.

        Args:
            kind: kind of blade to give indices for
            invert: whether to return all blades not of the kind

        Returns:
            indices of blades of a given kind in the algebra
        """
        return get_blade_of_kind_indices(self.blade_degrees, kind, self.max_degree, invert=invert)

    def get_blade_indices_of_degree(self, degree: int) -> tf.Tensor:
        """Find all indices of blades of the given degree.

        Args:
            degree: degree to return blades for

        Returns:
            indices of blades with the given degree in the algebra
        """
        return tf.gather(tf.range(self.num_blades), tf.where(self.blade_degrees == degree)[..., 0])

    def is_pure(self, tensor: tf.Tensor, blade_indices: tf.Tensor) -> bool:
        """Returns whether the given tensor is purely of the given blades
        and has no non-zero values for blades not in the given blades.

        Args:
            tensor: tensor to check purity for
            blade_indices: blade indices to check purity for

        Returns:
            Whether the tensor is purely of the given blades
            and has no non-zero values for blades not in the given blades
        """
        tensor = tf.convert_to_tensor(tensor, dtype_hint=tf.float32)
        blade_indices = tf.convert_to_tensor(
            blade_indices, dtype_hint=tf.int64)

        inverted_blade_indices = invert_blade_indices(
            self.num_blades, blade_indices)

        return tf.reduce_all(tf.gather(
            tensor,
            inverted_blade_indices,
            axis=-1
        ) == 0)

    def is_pure_kind(self, tensor: tf.Tensor, kind: BladeKind) -> bool:
        """Returns whether the given tensor is purely of a given kind
        and has no non-zero values for blades not of the kind.

        Args:
            tensor: tensor to check purity for
            kind: kind of blade to check purity for

        Returns:
            Whether the tensor is purely of a given kind
            and has no non-zero values for blades not of the kind
        """
        tensor = tf.convert_to_tensor(tensor, dtype_hint=tf.float32)
        inverted_kind_indices = self.get_kind_blade_indices(kind, invert=True)

        return tf.reduce_all(tf.gather(
            tensor,
            inverted_kind_indices,
            axis=-1
        ) == 0)

    def from_tensor(self, tensor: tf.Tensor, blade_indices: tf.Tensor) -> tf.Tensor:
        """Creates a geometric algebra tf.Tensor from a tf.Tensor and blade
        indices. The blade indices have to align with the last axis of the
        tensor.

        Args:
            tensor: tf.Tensor to take as values for the geometric algebra tensor
            blade_indices: Blade indices corresponding to the tensor. Can
            be obtained from blade names eg. using get_kind_blade_indices()
            or as indices from the blades list property.

        Returns:
            Geometric algebra tf.Tensor from tensor and blade indices
        """
        tensor = tf.convert_to_tensor(tensor, dtype_hint=tf.float32)

        # Put last axis on first axis so scatter_nd becomes easier.
        # Later undo the transposition again.
        t = tf.concat([[tensor.shape.ndims - 1],
                       tf.range(0, tensor.shape.ndims - 1)], axis=0)
        t_inv = tf.concat([tf.range(1, tensor.shape.ndims), [0]], axis=0)

        tensor = tf.transpose(tensor, t)

        tensor = tf.scatter_nd(tf.expand_dims(blade_indices, axis=-1), tensor,
                               [self.num_blades] + tensor.shape[1:])

        return tf.transpose(tensor, t_inv)

    def from_tensor_with_kind(self, tensor: tf.Tensor, kind: BladeKind) -> tf.Tensor:
        """Creates a geometric algebra tf.Tensor from a tf.Tensor and a kind.
        The kind's blade indices have to align with the last axis of the
        tensor.

        Args:
            tensor: tf.Tensor to take as values for the geometric algebra tensor
            kind: Kind corresponding to the tensor

        Returns:
            Geometric algebra tf.Tensor from tensor and kind
        """
        # Put last axis on first axis so scatter_nd becomes easier.
        # Later undo the transposition again.
        tensor = tf.convert_to_tensor(tensor, dtype_hint=tf.float32)
        kind_indices = self.get_kind_blade_indices(kind)
        return self.from_tensor(tensor, kind_indices)

    def from_scalar(self, scalar: numbers.Number) -> tf.Tensor:
        """Creates a geometric algebra tf.Tensor with scalar elements.

        Args:
            scalar: Elements to be used as scalars

        Returns:
            Geometric algebra tf.Tensor from scalars
        """
        return self.from_tensor_with_kind(tf.expand_dims(scalar, axis=-1), BladeKind.SCALAR)

    def e(self, *blades: List[str]) -> tf.Tensor:
        """Returns a geometric algebra tf.Tensor with the given blades set
        to 1.

        Args:
            blades: list of blade names, can be unnormalized

        Returns:
            tf.Tensor with blades set to 1
        """
        blade_signs, blade_indices = get_blade_indices_from_names(
            blades, self.blades)

        blade_indices = tf.convert_to_tensor(blade_indices)

        # Don't allow duplicate indices
        assert blade_indices.shape[0] == tf.unique(blade_indices)[0].shape[0]

        x = (
            tf.expand_dims(blade_signs, axis=-1) *
            tf.gather(self.blade_mvs, blade_indices)
        )

        # a, b -> b
        return tf.reduce_sum(x, axis=-2)

    def __getattr__(self, name: str) -> tf.Tensor:
        """Returns basis blade tensors if name was a basis."""
        if name.startswith("e") and (name[1:] == "" or int(name[1:]) >= 0):
            return self.e(name[1:])
        raise AttributeError

    def dual(self, tensor: tf.Tensor) -> tf.Tensor:
        """Returns the dual of the geometric algebra tensor.

        Args:
            tensor: Geometric algebra tensor to return dual for

        Returns:
            Dual of the geometric algebra tensor
        """
        tensor = tf.convert_to_tensor(tensor, dtype_hint=tf.float32)
        return self.dual_blade_signs * tf.gather(tensor, self.dual_blade_indices, axis=-1)

    def grade_automorphism(self, tensor: tf.Tensor) -> tf.Tensor:
        """Returns the geometric algebra tensor with odd grades negated.
        See https://en.wikipedia.org/wiki/Paravector#Grade_automorphism.

        Args:
            tensor: Geometric algebra tensor to return grade automorphism for

        Returns:
            Geometric algebra tensor with odd grades negated
        """
        tensor = tf.convert_to_tensor(tensor, dtype_hint=tf.float32)
        return mv_grade_automorphism(tensor, self.blade_degrees)

    def reversion(self, tensor: tf.Tensor) -> tf.Tensor:
        """Grade-reversion. See `reversion()`."""
        tensor = tf.convert_to_tensor(tensor, dtype_hint=tf.float32)
        return mv_reversion(tensor, self.blade_degrees)

    def conjugation(self, tensor: tf.Tensor) -> tf.Tensor:
        """Combines reversion and grade automorphism.
        See https://en.wikipedia.org/wiki/Paravector#Clifford_conjugation.

        Args:
            tensor: Geometric algebra tensor to return conjugate for

        Returns:
            Geometric algebra tensor after `reversion()` and `grade_automorphism()`
        """
        tensor = tf.convert_to_tensor(tensor, dtype_hint=tf.float32)
        return self.grade_automorphism(self.reversion(tensor))

    def inverse(self, a: tf.Tensor) -> tf.Tensor:
        """Returns the inverted geometric algebra tensor
        `X^-1` such that `X * X^-1 = 1` if
        it exists.

        Args:
            a: Geometric algebra tensor to return inverse for

        Returns:
            inverted geometric algebra tensor
        """
        a = tf.convert_to_tensor(a, dtype_hint=tf.float32)

        rev_a = self.reversion(a)
        divisor = self.geom_prod(a, rev_a)
        if not self.is_pure_kind(divisor, BladeKind.SCALAR):
            raise Exception(
                "Can't invert multi-vector (inversion divisor V ~V not scalar: %s)." % divisor)

        # Divide by scalar part
        return rev_a / divisor[..., :1]

    def reg_prod(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Returns the regressive product of two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the regressive product
            b: Geometric algebra tensor on the right hand side of
            the regressive product

        Returns:
            regressive product of a and b
        """
        a = tf.convert_to_tensor(a, dtype_hint=tf.float32)
        b = tf.convert_to_tensor(b, dtype_hint=tf.float32)

        return self.dual(self.ext_prod(self.dual(a), self.dual(b)))

    def ext_prod(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Returns the exterior product of two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the exterior product
            b: Geometric algebra tensor on the right hand side of
            the exterior product

        Returns:
            exterior product of a and b
        """
        a = tf.convert_to_tensor(a, dtype_hint=tf.float32)
        b = tf.convert_to_tensor(b, dtype_hint=tf.float32)

        return mv_multiply(a, b, self._cayley_outer)

    def geom_prod(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Returns the geometric product of two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the geometric product
            b: Geometric algebra tensor on the right hand side of
            the geometric product

        Returns:
            geometric product of a and b
        """
        a = tf.convert_to_tensor(a, dtype_hint=tf.float32)
        b = tf.convert_to_tensor(b, dtype_hint=tf.float32)

        a = tf.convert_to_tensor(a)
        b = tf.convert_to_tensor(b)
        return mv_multiply(a, b, self._cayley)

    def inner_prod(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Returns the inner product of two geometric
        algebra tensors.

        Args:
            a: Geometric algebra tensor on the left hand side of
            the inner product
            b: Geometric algebra tensor on the right hand side of
            the inner product

        Returns:
            inner product of a and b
        """
        a = tf.convert_to_tensor(a, dtype_hint=tf.float32)
        b = tf.convert_to_tensor(b, dtype_hint=tf.float32)

        return mv_multiply(a, b, self._cayley_inner)

    def mv_repr(self, a: tf.Tensor) -> str:
        """Returns a string representation for the given
        geometric algebra tensor.

        Args:
            a: Geometric algebra tensor to return the representation for

        Returns:
            string representation for `a`
        """
        a = tf.convert_to_tensor(a, dtype_hint=tf.float32)

        if len(a.shape) == 1:
            return "MultiVector[%s]" % " + ".join(
                "%.2f*%s" % (value, get_blade_repr(blade_name))
                for value, blade_name
                in zip(a, self.blades)
                if value != 0
            )
        else:
            return "MultiVector[batch_shape=%s]" % a.shape[:-1]

    def approx_exp(self, a: tf.Tensor, order: int = 50) -> tf.Tensor:
        """Returns an approximation of the exponential using a centered taylor series.

        Args:
            a: Geometric algebra tensor to return exponential for
            order: order of the approximation

        Returns:
            Approximation of `exp(a)`
        """
        a = tf.convert_to_tensor(a, dtype_hint=tf.float32)

        v = self.from_scalar(1.0)
        result = self.from_scalar(1.0)
        for i in range(1, order + 1):
            v = self.geom_prod(a, v)
            i_factorial = tf.exp(tf.math.lgamma(i + 1.0))
            result += v / i_factorial
        return result

    def approx_log(self, a: tf.Tensor, order: int = 50) -> tf.Tensor:
        """Returns an approximation of the natural logarithm using a centered
        taylor series. Only converges for multivectors where `||mv - 1|| < 1`.

        Args:
            a: Geometric algebra tensor to return logarithm for
            order: order of the approximation

        Returns:
            Approximation of `log(a)`
        """
        a = tf.convert_to_tensor(a, dtype_hint=tf.float32)

        result = self.from_scalar(0.0)

        a_minus_one = a - self.from_scalar(1.0)
        v = None

        for i in range(1, order + 1):
            v = a_minus_one if v is None else v * a_minus_one
            result += (((-1.0) ** i) / i) * v

        return -result

    def int_pow(self, a: tf.Tensor, n: int) -> tf.Tensor:
        """Returns the geometric algebra tensor to the power of an integer
        using repeated multiplication.

        Args:
            a: Geometric algebra tensor to raise
            n: integer power to raise the multivector to

        Returns:
            `a` to the power of `n`
        """
        a = tf.convert_to_tensor(a, dtype_hint=tf.float32)

        if not isinstance(n, int):
            raise Exception("n must be an integer.")
        if n < 0:
            raise Exception("Can't raise to negative powers.")

        if n == 0:
            # TODO: more efficient (ones only in scalar)
            return tf.ones_like(a) * self.e("")

        result = a
        for i in range(n - 1):
            result = self.geom_prod(result, a)
        return result

    def keep_blades(self, a: tf.Tensor, blade_names: Union[List[str], str]) -> tf.Tensor:
        """Takes a geometric algebra tensor and returns it with only the given
        blades as non-zeros.

        Args:
            a: Geometric algebra tensor to copy
            blade_names: Blades to keep

        Returns:
            `a` with only `blade_names` elements as non-zeros
        """
        a = tf.convert_to_tensor(a, dtype_hint=tf.float32)

        if isinstance(blade_names, str):
            blade_names = [blade_names]

        _, blade_indices = get_blade_indices_from_names(
            blade_names, self.blades)

        blade_values = tf.gather(a, blade_indices, axis=-1)

        return self.from_tensor(blade_values, blade_indices)

    def select_blades(self, a: tf.Tensor, blade_names: Union[List[str], str]) -> tf.Tensor:
        """Takes a geometric algebra tensor and returns a `tf.Tensor` with the
        blades in blade_names on the last axis.


        Args:
            a: Geometric algebra tensor to copy
            blade_names: Blades to keep

        Returns:
            `tf.Tensor` based on `a` with `blade_names` on last axis.
        """
        a = tf.convert_to_tensor(a, dtype_hint=tf.float32)

        is_single_blade = isinstance(blade_names, str)
        if is_single_blade:
            blade_names = [blade_names]

        blade_signs, blade_indices = get_blade_indices_from_names(
            blade_names, self.blades)

        result = blade_signs * tf.gather(a, blade_indices, axis=-1)

        if is_single_blade:
            return result[..., 0]

        return result

    def __call__(self, a: tf.Tensor) -> MultiVector:
        """Creates a `MultiVector` from a geometric algebra tensor.
        Mainly used as a wrapper for the algebra's functions for convenience.

        Args:
            a: Geometric algebra tensor to return `MultiVector` for

        Returns:
            `MultiVector` for `a`
        """
        return MultiVector(tf.convert_to_tensor(a), self)
