from __future__ import annotations
from typing import Union


class MultiVector:
    def __init__(self, blade_values, algebra):
        self._blade_values = blade_values
        self._algebra = algebra
        self._n = -1

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
        assert self._blade_values.shape.ndims > 1

        self._n = 0
        return self

    def __next__(self) -> self:
        if self._n <= self._blade_values.shape[0]:
            return MultiVector(
                self._blade_values[self._n],
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
