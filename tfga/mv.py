"""Defines the `MultiVector` class which is used as a convenience wrapper
for `GeometricAlgebra` operations.
"""
from __future__ import annotations
from typing import Union

from .blades import BladeKind


class MultiVector:
    """Wrapper for geometric algebra tensors using `GeometricAlgebra`
    operations in a less verbose way using operators.
    """

    def __init__(self, blade_values: tf.Tensor, algebra: GeometricAlgebra):
        """Initializes a MultiVector from a geometric algebra `tf.Tensor`
        and its corresponding `GeometricAlgebra`.

        Args:
            blade_values: Geometric algebra `tf.Tensor` with as many elements
            on its last axis as blades in the algebra
            algebra: `GeometricAlgebra` instance corresponding to the geometric
            algebra tensor
        """

        self._blade_values = blade_values
        self._algebra = algebra
        self._n = -1  # used for iteration (__iter__, __next__)

    @property
    def tensor(self):
        """Geometric algebra tensor holding the values of this multivector."""
        return self._blade_values

    @property
    def algebra(self):
        """`GeometricAlgebra` instance this multivector belongs to."""
        return self._algebra

    @property
    def batch_shape(self):
        """Batch shape of the multivector (ie. the shape of all axes except
        for the last one in the geometric algebra tensor).
        """
        return self._blade_values.shape[:-1]

    def __len__(self) -> int:
        """Number of elements on the first axis of the geometric algebra
        tensor."""
        return self._blade_values.shape[0]

    def __iter__(self) -> self:

        self._n = 0
        return self

    def __next__(self) -> self:
        n = self._n
        if n <= self._blade_values.shape[0]:
            self._n += 1

            # If we only have one axis left, return the
            # actual numbers, otherwise return a new
            # multivector.
            if self._blade_values.shape.ndims == 1:
                return self._blade_values[n]
            else:
                return MultiVector(
                    self._blade_values[n],
                    self._algebra
                )
        raise StopIteration

    def __xor__(self, other: self) -> self:
        """Exterior product. See `GeometricAlgebra.ext_prod()`"""
        assert isinstance(other, MultiVector)

        return MultiVector(
            self._algebra.ext_prod(self._blade_values, other._blade_values),
            self._algebra
        )

    def __or__(self, other: self) -> self:
        """Inner product. See `GeometricAlgebra.inner_prod()`"""
        assert isinstance(other, MultiVector)

        return MultiVector(
            self._algebra.inner_prod(self._blade_values, other._blade_values),
            self._algebra
        )

    def __mul__(self, other: self) -> self:
        """Geometric product. See `GeometricAlgebra.geom_prod()`"""
        assert isinstance(other, MultiVector)

        return MultiVector(
            self._algebra.geom_prod(self._blade_values, other._blade_values),
            self._algebra
        )

    def __and__(self, other: self) -> self:
        """Regressive product. See `GeometricAlgebra.reg_prod()`"""
        assert isinstance(other, MultiVector)

        return MultiVector(
            self._algebra.reg_prod(self._blade_values, other._blade_values),
            self._algebra
        )

    def __invert__(self) -> self:
        """Reversion. See `GeometricAlgebra.reversion()`"""
        return MultiVector(
            self._algebra.reversion(self._blade_values),
            self._algebra
        )

    def __negate__(self) -> self:
        """Negation."""
        return MultiVector(
            -self._blade_values,
            self._algebra
        )

    def __add__(self, other: self) -> self:
        """Addition of multivectors."""
        assert isinstance(other, MultiVector)

        return MultiVector(
            self._blade_values + other._blade_values,
            self._algebra
        )

    def __sub__(self, other: self) -> self:
        """Subtraction of multivectors."""
        assert isinstance(other, MultiVector)

        return MultiVector(
            self._blade_values - other._blade_values,
            self._algebra
        )

    def __pow__(self, n: int) -> self:
        """Multivector raised to an integer power."""
        return MultiVector(
            self._algebra.int_pow(self._blade_values, n),
            self._algebra
        )

    def __getitem__(self, key: Union[str, List[str]]) -> self:
        """`MultiVector` with only passed blades as non-zeros."""
        return MultiVector(
            self._algebra.keep_blades(self._blade_values, key),
            self._algebra
        )

    def __call__(self, key: Union[str, List[str]]):
        """`tf.Tensor` with passed blades on last axis."""
        return self._algebra.select_blades(self._blade_values, key)

    def __repr__(self) -> str:
        return self._algebra.mv_repr(self._blade_values)

    def inverse(self) -> self:
        """Inverse. See `GeometricAlgebra.inverse()`."""
        return MultiVector(
            self._algebra.inverse(self._blade_values),
            self._algebra
        )

    def dual(self) -> self:
        """Dual. See `GeometricAlgebra.dual()`."""
        return MultiVector(
            self._algebra.dual(self._blade_values),
            self._algebra
        )

    def conjugation(self) -> self:
        """Conjugation. See `GeometricAlgebra.conjugation()`."""
        return MultiVector(
            self._algebra.conjugation(self._blade_values),
            self._algebra
        )

    def grade_automorphism(self) -> self:
        """Grade automorphism. See `GeometricAlgebra.grade_automorphism()`."""
        return MultiVector(
            self._algebra.grade_automorphism(self._blade_values),
            self._algebra
        )

    def approx_exp(self) -> self:
        """Approximate exponential. See `GeometricAlgebra.approx_exp()`."""
        return MultiVector(
            self._algebra.approx_exp(self._blade_values),
            self._algebra
        )

    def approx_log(self) -> self:
        """Approximate logarithm. See `GeometricAlgebra.approx_log()`."""
        return MultiVector(
            self._algebra.approx_log(self._blade_values),
            self._algebra
        )

    def is_pure_kind(self, kind: BladeKind) -> bool:
        """Whether the `MultiVector` is of a pure kind."""
        return self._algebra.is_pure_kind(self._blade_values, kind=kind)
