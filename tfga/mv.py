from __future__ import annotations
from typing import Union

from .blades import BladeKind


class MultiVector:
    def __init__(self, blade_values, algebra):
        self._blade_values = blade_values
        self._algebra = algebra
        self._n = -1 # used for iteration (__iter__, __next__)

    @property
    def tensor(self):
        return self._blade_values

    @property
    def algebra(self):
        return self._algebra

    @property
    def batch_shape(self):
        return self._blade_values.shape[:-1]

    def __len__(self) -> int:
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
        assert isinstance(other, MultiVector)

        return MultiVector(
            self._algebra.ext_prod(self._blade_values, other._blade_values),
            self._algebra
        )

    def __or__(self, other: self) -> self:
        assert isinstance(other, MultiVector)

        return MultiVector(
            self._algebra.inner_prod(self._blade_values, other._blade_values),
            self._algebra
        )

    def __mul__(self, other: self) -> self:
        assert isinstance(other, MultiVector)

        return MultiVector(
            self._algebra.geom_prod(self._blade_values, other._blade_values),
            self._algebra
        )

    def __and__(self, other: self) -> self:
        assert isinstance(other, MultiVector)

        return MultiVector(
            self._algebra.reg_prod(self._blade_values, other._blade_values),
            self._algebra
        )

    def __invert__(self) -> self:
        return MultiVector(
            self._algebra.reversion(self._blade_values),
            self._algebra
        )

    def __negate__(self) -> self:
        return MultiVector(
            -self._blade_values,
            self._algebra
        )

    def __add__(self, other: self) -> self:
        assert isinstance(other, MultiVector)

        return MultiVector(
            self._blade_values + other._blade_values,
            self._algebra
        )

    def __sub__(self, other: self) -> self:
        assert isinstance(other, MultiVector)

        return MultiVector(
            self._blade_values - other._blade_values,
            self._algebra
        )

    def __pow__(self, n: int) -> self:
        return MultiVector(
            self._algebra.int_pow(self._blade_values, n),
            self._algebra
        )

    def __getitem__(self, key: Union[str, List[str]]) -> self:
        return MultiVector(
            self._algebra.keep_blades(self._blade_values, key),
            self._algebra
        )

    def __call__(self, key: Union[str, List[str]]):
        return self._algebra.select_blades(self._blade_values, key)

    def __repr__(self) -> str:
        return self._algebra.mv_repr(self._blade_values)

    def inverse(self) -> self:
        return MultiVector(
            self._algebra.inverse(self._blade_values),
            self._algebra
        )

    def dual(self) -> self:
        return MultiVector(
            self._algebra.dual(self._blade_values),
            self._algebra
        )

    def conjugation(self) -> self:
        return MultiVector(
            self._algebra.conjugation(self._blade_values),
            self._algebra
        )

    def grade_automorphism(self) -> self:
        return MultiVector(
            self._algebra.grade_automorphism(self._blade_values),
            self._algebra
        )

    def approx_exp(self) -> self:
        return MultiVector(
            self._algebra.approx_exp(self._blade_values),
            self._algebra
        )

    def approx_log(self) -> self:
        return MultiVector(
            self._algebra.approx_log(self._blade_values),
            self._algebra
        )

    def is_pure_kind(self, kind: BladeKind) -> bool:
        return self._algebra.is_pure_kind(self._blade_values, kind=kind)
