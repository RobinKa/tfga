"""Provides classes and operations for performing MultiVector algebra
with TensorFlow.

The `GeometricAlgebra` class is used to construct the algebra given a metric.
After that new multivectors of type `MultiVector` can be created from it
that can be multiplied and added together among other operations.
"""
from typing import List, Any, Union
import numbers
import tensorflow as tf
import numpy as np

from .cayley import get_cayley_tensor, blades_from_bases
from .blades import BladeKind, get_blade_of_kind_indices, get_blade_indices_from_names, get_blade_repr
from .mv_ops import mv_multiply, mv_reversion, mv_grade_automorphism


class GeometricAlgebra:
    """Class used as the basis for creating multi-vectors.
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
        def _is_mv(arg):
            return isinstance(arg, tf.Tensor) and arg.shape.ndims > 0 and arg.shape[-1] == self.num_blades
        new_args = [self.mv_repr(arg) if _is_mv(arg) else arg for arg in args]

        print(*new_args, **kwargs)

    @property
    def metric(self) -> tf.Tensor:
        """Metric list which contains the number that each
        basis vector in the algebra squares to."""
        return self._metric

    @property
    def cayley(self) -> tf.Tensor:
        """`MxMxM` tensor where `M` is the number of basis
        vectors in the algebra. Used for calculating the
        geometric product:

        `a * b = b @ (a @ cayley)`
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
        """List of all blade multivectors in the algebra."""
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
        """List of basis vectors as multivectors."""
        return self._basis_mvs

    def get_kind_blade_indices(self, kind: BladeKind, invert: bool = False) -> tf.Tensor:
        """Find all indices of blades of a given kind in the algebra.

        Args:
            kind: kind of blade to give indices for

        Returns:
            indices of blades of a given kind in the algebra
        """
        return get_blade_of_kind_indices(self.blade_degrees, kind, self.max_degree, invert=invert)

    def is_pure_kind(self, tensor: tf.Tensor, kind: BladeKind) -> bool:
        """Returns whether this multivector is purely of a given kind
        and has no non-zero values for blades not of the kind.

        Args:
            kind: kind of blade to check purity for

        Returns:
            Whether this multivector is purely of a given kind
            and has no non-zero values for blades not of the kind
        """
        kind_indices = self.get_kind_blade_indices(kind, invert=True)
        return tf.reduce_all(tf.gather(
            tensor,
            kind_indices,
            axis=-1
        ) == 0)

    def from_tensor(self, tensor: tf.Tensor, blade_indices: tf.Tensor) -> tf.Tensor:
        """Creates a multivector from a tf.Tensor and blade indices.
        The blade indices have to align with the last axis of the tensor.

        Args:
            tensor: tf.Tensor to take as values for a multivector
            blade_indices: Blade indices corresponding to the tensor. Can
            be obtained from blade names eg. using get_kind_blade_indices()
            or as indices from the blades list property.

        Returns:
            MultiVector from tensor and blade indices
        """
        # Put last axis on first axis so scatter_nd becomes easier.
        # Later undo the transposition again.
        t = tf.concat([[tensor.shape.ndims - 1],
                       tf.range(0, tensor.shape.ndims - 1)], axis=0)
        t_inv = tf.concat([tf.range(1, tensor.shape.ndims), [0]], axis=0)

        tensor = tf.transpose(tensor, t)

        tensor = tf.scatter_nd(tf.expand_dims(blade_indices, axis=-1), tensor,
                               [self.num_blades] + tensor.shape[1:])

        return tf.transpose(tensor, t_inv)

    def from_tensor_with_kind(self, tensor, kind):
        # Put last axis on first axis so scatter_nd becomes easier.
        # Later undo the transposition again.
        kind_indices = self.get_kind_blade_indices(kind)
        return self.from_tensor(tensor, kind_indices)

    def from_scalar(self, scalar):
        return self.from_tensor_with_kind(tf.expand_dims(scalar, axis=-1), BladeKind.SCALAR)

    def e(self, *blades: List[str]) -> tf.Tensor:
        """Returns a multivector with the given blades set to 1.

        Args:
            blades: list of blade names, can be unnormalized

        Returns:
            MultiVector with blades set to 1
        """
        blade_signs, blade_indices = get_blade_indices_from_names(
            blades, self.blades)

        blade_indices = tf.convert_to_tensor(blade_indices)

        # Don't allow duplicate indices
        no_duplicates_assertion = tf.Assert(
            blade_indices.shape[0] == tf.unique(blade_indices)[0].shape[0],
            [blade_indices]
        )

        with tf.control_dependencies([no_duplicates_assertion]):
            x = (tf.expand_dims(blade_signs, axis=-1) *
                 tf.gather(self.blade_mvs, blade_indices)
                 )

            # a, b -> b
            return tf.reduce_sum(x, axis=-2)

    def dual(self, tensor: tf.Tensor) -> tf.Tensor:
        """Returns the dual of the MultiVector.

        Returns:
            Dual of the MultiVector
        """
        return self.dual_blade_signs * tf.gather(tensor, self.dual_blade_indices, axis=-1)

    def grade_automorphism(self, tensor: tf.Tensor) -> tf.Tensor:
        """Returns the multivector with odd grades negated.
        See https://en.wikipedia.org/wiki/Paravector#Grade_automorphism.

        Returns:
            `MultiVector` with odd grades negated
        """
        return mv_grade_automorphism(tensor, self.blade_degrees)

    def reversion(self, tensor: tf.Tensor) -> tf.Tensor:
        """Grade-reversion. See `reversion()`."""
        return mv_reversion(tensor, self.blade_degrees)

    def conjugation(self, tensor: tf.Tensor) -> tf.Tensor:
        """Combines reversion and grade automorphism.
        See https://en.wikipedia.org/wiki/Paravector#Clifford_conjugation.

        Returns:
            `MultiVector` after `reversion()` and `grade_automorphism()`
        """
        return self.grade_automorphism(self.reversion(tensor))

    def inverse(self, a: tf.Tensor) -> tf.Tensor:
        """Returns the inverted multivector
        `X^-1` such that `X * X^-1 = 1` if
        it exists.

        Returns:
            inverted `MultiVector`
        """
        rev_a = self.reversion(a)
        divisor = self.geom_prod(a, rev_a)
        if not self.is_pure_kind(divisor, BladeKind.SCALAR):
            raise Exception(
                "Can't invert multi-vector (inversion divisor V ~V not scalar: %s)." % divisor)

        # Divide by scalar part
        return rev_a / divisor[..., :1]

    def reg_prod(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Returns the regressive product of the multivector and
        another multivector.

        Args:
            other: object to take regressive product with

        Returns:
            regressive product of multivector and other
        """
        return self.dual(self.ext_prod(self.dual(a), self.dual(b)))

    def ext_prod(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Returns the exterior product.

        Args:
            other: object to calculate exterior product with

        Returns:
            exterior product of `self` and `other`
        """
        return mv_multiply(a, b, self._cayley_outer)

    def geom_prod(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Returns the geometric product.

        Args:
            other: object to multiply with

        Returns:
            geometric product of `self` and `other`
        """
        a = tf.convert_to_tensor(a)
        b = tf.convert_to_tensor(b)
        return mv_multiply(a, b, self._cayley)

    def inner_prod(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Returns the inner product.

        Args:
            other: object to calculate inner product with

        Returns:
            inner product of `self` and `other`
        """
        return mv_multiply(a, b, self._cayley_inner)

    def mv_repr(self, a: tf.Tensor) -> str:
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
            order: order of the approximation

        Returns:
            Approximation of `exp(self)`
        """
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
            order: order of the approximation

        Returns:
            Approximation of `log(self)`
        """
        result = self.from_scalar(0.0)

        a_minus_one = a - self.from_scalar(1.0)
        v = None

        for i in range(1, order + 1):
            v = a_minus_one if v is None else v * a_minus_one
            result += (((-1.0) ** i) / i) * v

        return -result

    def int_pow(self, a: tf.Tensor, n: int) -> tf.Tensor:
        """Returns the multivector to the power of an integer
        using repeated multiplication.

        Args:
            n: integer power to raise the multivector to

        Returns:
            `MultiVector` to the power of `n`
        """
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
        if isinstance(blade_names, str):
            blade_names = [blade_names]

        _, blade_indices = get_blade_indices_from_names(
            blade_names, self.blades)

        blade_values = tf.gather(a, blade_indices, axis=-1)

        return self.from_tensor(blade_values, blade_indices)

    def select_blades(self, a: tf.Tensor, blade_names: Union[List[str], str]) -> tf.Tensor:
        is_single_blade = isinstance(blade_names, str)
        if is_single_blade:
            blade_names = [blade_names]

        blade_signs, blade_indices = get_blade_indices_from_names(
            blade_names, self.blades)

        result = blade_signs * tf.gather(a, blade_indices, axis=-1)

        if is_single_blade:
            return result[..., 0]

        return result