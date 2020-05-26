"""Provides classes and operations for performing MultiVector algebra
with TensorFlow.

The `GeometricAlgebra` class is used to construct the algebra given a metric.
After that new multivectors of type `MultiVector` can be created from it
that can be multiplied and added together among other operations.
"""
from typing import List, Any
import numbers
import tensorflow as tf
import numpy as np

from .cayley import get_cayley_tensor, blades_from_bases
from .mv import MultiVector
from .blades import BladeKind, get_blade_of_kind_indices


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

        self._basis_mvs = [
            MultiVector(
                blade_values=tf.convert_to_tensor([1], dtype=tf.float32),
                blade_indices=tf.convert_to_tensor([i+1], dtype=tf.int64),
                algebra=self
            )
            for i in range(self._num_bases)
        ]

        self._blade_mvs = [
            MultiVector(
                blade_values=tf.convert_to_tensor([1], dtype=tf.float32),
                blade_indices=[i],
                algebra=self
            )
            for i in range(self._num_blades)
        ]

        # Find the dual by looking at the anti-diagonal in the Cayley tensor.
        self._dual_blade_indices = []
        self._dual_blade_signs = []

        for blade_index in range(self._num_blades):
            dual_index = self.num_blades - blade_index - 1
            anti_diag = self._cayley[blade_index, dual_index]
            dual_sign = tf.gather(anti_diag, tf.where(anti_diag != 0.0)[..., 0])[..., 0]
            self._dual_blade_indices.append(dual_index)
            self._dual_blade_signs.append(dual_sign)

        self._dual_blade_indices = tf.convert_to_tensor(
            self._dual_blade_indices, dtype=tf.int64)
        self._dual_blade_signs = tf.convert_to_tensor(
            self._dual_blade_signs, dtype=tf.float32)

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

        `a * b = a @ (b @ cayley)`
        """
        return self._cayley

    @property
    def cayley_inner(self) -> tf.Tensor:
        """`Analagous to cayley but for inner product."""
        return self._cayley_inner

    @property
    def cayley_outer(self) -> tf.Tensor:
        """`Analagous to cayley but for outer product."""
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
    def blade_mvs(self) -> List[MultiVector]:
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
    def basis_mvs(self) -> List[MultiVector]:
        """List of basis vectors as multivectors."""
        return self._basis_mvs

    def get_kind_blade_indices(self, kind: BladeKind) -> tf.Tensor:
        """Find all indices of blades of a given kind in the algebra.

        Args:
            kind: kind of blade to give indices for

        Returns:
            indices of blades of a given kind in the algebra
        """

        return get_blade_of_kind_indices(self.blade_degrees, kind, self.max_degree)

    def zeros(self, batch_shape: List[int], kind: BladeKind = BladeKind.MV,
              dtype: tf.DType = tf.float32) -> MultiVector:
        """Returns a multivector of the algebra of a given kind with all its
        blades set to zero.

        Args:
            batch_shape: shape for how many multivectors to return
            kind: kind of multivector to create
            dtype: data type of the underlying multivector data

        Returns:
            `MultiVector` of the algebra of `kind` with all its
            blades set to zero
        """
        blade_indices = self.get_kind_blade_indices(kind)
        mv = tf.zeros([*batch_shape, len(blade_indices)], dtype=dtype)
        return MultiVector(blade_values=mv, blade_indices=blade_indices, algebra=self)

    def ones(self, batch_shape: List[int], kind: BladeKind = BladeKind.MV,
             dtype: tf.DType = tf.float32) -> MultiVector:
        """Returns a multivector of the algebra of a given kind with all its
        blades set to one

        Args:
            batch_shape: shape for how many multivectors to return
            kind: kind of multivector to create
            dtype: data type of the underlying multivector data

        Returns:
            `MultiVector` of the algebra of `kind` with all its
            blades set to one
        """
        blade_indices = self.get_kind_blade_indices(kind)
        mv = tf.ones([*batch_shape, len(blade_indices)], dtype=dtype)
        return MultiVector(blade_values=mv, blade_indices=blade_indices, algebra=self)

    def fill(self, batch_shape: List[int], fill_value: numbers.Number,
             kind=BladeKind.MV) -> MultiVector:
        """Returns a multivector of the algebra of a given kind with all its
        blades set to the fill value. The dtype of the fill value will be used
        as the dtype for the multivector.

        Args:
            batch_shape: shape for how many multivectors to return
            fill_value: value to use for all blades of `kind`
            kind: kind of multivector to create

        Returns:
            `MultiVector` of the algebra of `kind` with all its
            blades set to `fill_value`
        """
        blade_indices = self.get_kind_blade_indices(kind)
        mv = tf.fill([*batch_shape, len(blade_indices)], fill_value)
        return MultiVector(blade_values=mv, blade_indices=blade_indices, algebra=self)

    def as_mv(self, x: Any) -> MultiVector:
        """Converts any input to a `MultiVector` if possible.

        Args:
            x: input to convert to a `MultiVector`

        Returns:
            `x` converted to a MultiVector
        """
        if isinstance(x, MultiVector):
            return x
        elif isinstance(x, numbers.Number):
            return self.fill([], fill_value=x, kind="scalar")
        elif ((isinstance(x, tf.Tensor) or isinstance(x, np.ndarray)) and
              (len(x.shape) == 0 or (len(x.shape) == 1 and x.shape[0] == 1))):
            return self.fill([], fill_value=x, kind="scalar")
        raise Exception(
            "Can't convert argument of type %s to multi-vector." % type(x))
