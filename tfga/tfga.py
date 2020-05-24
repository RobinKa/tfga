import tensorflow as tf
import numbers
import numpy as np
from .cayley import get_cayley_tensor, blades_from_bases


def get_blade_repr(blade_name):
    if blade_name == "":
        return "1"
    return "e_%s" % blade_name


def is_blade_kind(blade_degrees, kind, max_degree):
    if kind == "mv":
        return tf.constant(True, shape=[len(blade_degrees)])
    elif kind == "even":
        return blade_degrees % 2 == 0
    elif kind == "odd":
        return blade_degrees % 2 == 1
    elif kind == "scalar":
        return blade_degrees == 0
    elif kind == "pseudoscalar":
        return blade_degrees == max_degree
    elif kind == "vector":
        return blade_degrees == 1
    elif kind == "pseudovector":
        return blade_degrees == max_degree - 1
    raise Exception("Unknown blade kind: %s" % kind)


def get_blade_of_kind_indices(blade_degrees, kind, max_degree, invert=False):
    cond = is_blade_kind(blade_degrees, kind, max_degree)
    cond = tf.math.logical_xor(cond, invert)
    return tf.where(cond)[:, 0]


class GeometricAlgebra:
    def __init__(self, metric):
        self._metric = tf.convert_to_tensor(metric, dtype=tf.float32)

        self._num_bases = len(metric)
        self._bases = list(map(str, range(self._num_bases)))

        self._blades, self._blade_degrees = blades_from_bases(self._bases)
        self._blade_degrees = tf.convert_to_tensor(self._blade_degrees)
        self._num_blades = len(self._blades)

        # [Blades, Blades, Blades]
        self._cayley = tf.convert_to_tensor(
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

    @property
    def metric(self):
        return self._metric

    @property
    def cayley(self):
        return self._cayley

    @property
    def blades(self):
        return self._blades

    @property
    def num_blades(self):
        return self._num_blades

    @property
    def blade_degrees(self):
        return self._blade_degrees

    @property
    def max_degree(self):
        return self._num_blades - 1

    @property
    def basis_mvs(self):
        return self._basis_mvs

    def get_kind_blade_indices(self, kind):
        return get_blade_of_kind_indices(self.blade_degrees, kind, self.max_degree)

    def zeros(self, batch_shape, kind="mv", dtype=tf.float32, name=None):
        blade_indices = self.get_kind_blade_indices(kind)
        mv = tf.zeros([*batch_shape, len(blade_indices)],
                      dtype=dtype, name=name)
        return MultiVector(blade_values=mv, blade_indices=blade_indices, algebra=self)

    def ones(self, batch_shape, kind="mv", dtype=tf.float32, name=None):
        blade_indices = self.get_kind_blade_indices(kind)
        mv = tf.ones([*batch_shape, len(blade_indices)],
                     dtype=dtype, name=name)
        return MultiVector(blade_values=mv, blade_indices=blade_indices, algebra=self)

    def fill(self, batch_shape, fill_value, kind="mv", name=None):
        blade_indices = self.get_kind_blade_indices(kind)
        mv = tf.fill([*batch_shape, len(blade_indices)], fill_value,
                     name=name)
        return MultiVector(blade_values=mv, blade_indices=blade_indices, algebra=self)

    def as_mv(self, x):
        if isinstance(x, MultiVector):
            return x
        elif isinstance(x, numbers.Number):
            return self.fill([], fill_value=x, kind="scalar")
        elif isinstance(x, tf.Tensor) and len(x.shape) == 0 or (len(x.shape == 1) and x.shape[0] == 1):
            return self.fill([], fill_value=x, kind="scalar")
        raise Exception("Can't convert argument to multi-vector.")


class MultiVector:
    def __init__(self, blade_values, blade_indices, algebra):
        # blade_values: [*Batch, len(blade_indices)]
        self._blade_indices = blade_indices
        self._blade_values = blade_values
        self._algebra = algebra

    @property
    def blade_values(self):
        return self._blade_values

    @property
    def blade_indices(self):
        return self._blade_indices

    @property
    def algebra(self):
        return self._algebra

    def with_changes(self, blade_values=None, blade_indices=None):
        return MultiVector(
            blade_values=self.blade_values if blade_values is None else blade_values,
            blade_indices=self.blade_indices if blade_indices is None else blade_indices,
            algebra=self.algebra
        )

    def get_kind_indices(self, kind, invert=False):
        blade_degrees = tf.gather(
            self.algebra.blade_degrees, self.blade_indices
        )

        return get_blade_of_kind_indices(blade_degrees, kind, self.algebra.max_degree, invert=invert)

    def get_part_mv(self, kind):
        kind_indices = self.get_kind_indices(kind)

        result_indices = tf.gather(self.blade_indices, kind_indices, axis=-1)
        result_values = tf.gather(self.blade_values, kind_indices, axis=-1)

        return self.with_changes(
            blade_values=result_values,
            blade_indices=result_indices
        )

    def get_part(self, kind):
        kind_indices = self.get_kind_indices(kind)
        result_values = tf.gather(self.blade_values, kind_indices, axis=-1)
        return result_values

    def is_pure_kind(self, kind):
        other_kind_indices = self.get_kind_indices(kind, invert=True)
        return tf.reduce_all(tf.gather(self.blade_values, other_kind_indices, axis=-1) == 0)

    @property
    def scalar(self):
        return self.get_part("scalar")

    @property
    def scalar_mv(self):
        return self.get_part_mv("scalar")

    @property
    def is_pure_scalar(self):
        return self.is_pure_kind("scalar")

    @property
    def batch_shape(self):
        return self.blade_values.shape[:-1]

    @property
    def values_shape(self):
        return self.blade_values.shape[-1:]

    def reversion(self):
        """Grade-reversion. See https://en.wikipedia.org/wiki/Paravector#Reversion_conjugation."""
        blade_degrees = tf.cast(
            tf.gather(self.algebra.blade_degrees, self.blade_indices), tf.float32)

        # for each blade, 0 if even number of swaps required, else 1
        odd_swaps = tf.cast(
            tf.floor(blade_degrees * (blade_degrees - 0.5)) % 2, tf.float32)

        # [0, 1] -> [-1, 1]
        reversion_signs = 1.0 - 2.0 * odd_swaps

        return self.with_changes(
            blade_values=reversion_signs * self.blade_values
        )

    def grade_automorphism(self):
        """Negates odd grades. See https://en.wikipedia.org/wiki/Paravector#Grade_automorphism."""
        blade_degrees = tf.cast(
            tf.gather(self.algebra.blade_degrees, self.blade_indices), tf.float32)
        signs = 1.0 - 2.0 * (blade_degrees % 2)
        return self.with_changes(
            blade_values=signs * self.blade_values
        )

    def conjugation(self):
        """Combines reversion and grade automorphism.
        See https://en.wikipedia.org/wiki/Paravector#Clifford_conjugation.
        """
        return self.reversion().grade_automorphism()

    def __invert__(self):
        """Grade-reversion."""
        return self.reversion()

    def __truediv__(self, other):
        """Division of a multi-vector by a scalar."""
        other = self.algebra.as_mv(other)

        if other.is_pure_scalar:
            divisor = other.scalar
            return self.with_changes(blade_values=self.blade_values / divisor)

        raise Exception(
            "Division of two multi-vectors is ambiguous (left or right inverse?). Use a.inverse() * b or b * a.inverse() instead.")

    def inverse(self):
        """X^-1 such that X * X^-1 = 1"""
        rev_self = ~self
        divisor = self * rev_self
        if not divisor.is_pure_scalar:
            raise Exception(
                "Can't invert multi-vector (inversion divisor V ~V not scalar: %s)." % divisor)
        return rev_self / divisor

    def __mul__(self, other):
        """Geometric product."""
        other = self.algebra.as_mv(other)

        sub_cayley = tf.gather(self._algebra.cayley,
                               self.blade_indices, axis=0)
        sub_cayley = tf.gather(sub_cayley, other.blade_indices, axis=1)

        # [N, 3]
        x = tf.where(sub_cayley != 0)[:, 2]
        result_blade_indices, _ = tf.unique(x, tf.int64)
        sub_cayley = tf.gather(sub_cayley, result_blade_indices, axis=2)

        result_blade_values = tf.einsum("...i,...j,...ijk->...k",
                                        self._blade_values,
                                        other.blade_values,
                                        sub_cayley)

        return self.with_changes(
            blade_values=result_blade_values,
            blade_indices=result_blade_indices
        )

    def __rmul__(self, other):
        return self * other

    def __or__(self, other):
        """Inner product."""
        other = self.algebra.as_mv(other)
        return 0.5 * (self * other + other * self)

    def __ror__(self, other):
        return self | other

    def __xor__(self, other):
        """Exterior product."""
        other = self.algebra.as_mv(other)
        return 0.5 * (self * other - other * self)

    def __rxor__(self, other):
        return self ^ other

    def __eq__(self, other):
        other = self.algebra.as_mv(other)

        # TODO: Make sure blade indices are broadcastable.

        # Align self and other indices (if possible)
        # to the compare all indices and values.
        reindex_a = tf.argsort(self.blade_indices)
        reindex_b = tf.argsort(other.blade_indices)

        sorted_ind_a = tf.gather(self.blade_indices, reindex_a, axis=-1)
        sorted_ind_b = tf.gather(other.blade_indices, reindex_b, axis=-1)

        sorted_val_a = tf.gather(self.blade_values, reindex_a, axis=-1)
        sorted_val_b = tf.gather(other.blade_values, reindex_b, axis=-1)

        return tf.reduce_all(sorted_ind_a == sorted_ind_b) and tf.reduce_all(sorted_val_a == sorted_val_b)

    def __add__(self, other):
        other = self.algebra.as_mv(other)

        # vals: [20, 21, 22] [23, 24, 25]
        # ind: [1, 2, 3], [3, 2, 10]

        # concat indices: [1, 2, 3, 3, 2, 10]
        concat_indices = tf.concat(
            [self.blade_indices, other.blade_indices],
            axis=0
        )

        # new_indices: [1, 2, 3, 10]
        # remapped_indices (index of old values in new_indices): [0, 1, 2, 2, 1, 3]
        new_indices, remapped_indices = tf.unique(concat_indices, tf.int64)

        # concat values: [20, 21, 22, 23, 24, 25]
        concat_values = tf.concat(
            [self.blade_values, other.blade_values],
            axis=len(self.batch_shape)
        )

        # data: [20, 21, 22, 23, 24, 25]
        # segment_ids: [0, 1, 2, 2, 1, 3]
        # result: [20, 21+24, 22+23, 25]

        # unsorted_segment_sum only works on the first index so we need to transpose
        # our values on the last index to the first index and later untranspose again.

        # range: [0, 1, 2, 3]
        # t: [3, 0, 1, 2]
        # t^-1: [1, 2, 3, 0]
        transpose_indices = tf.roll(
            tf.range(len(concat_values.shape)), shift=1, axis=0)
        untranspose_indices = tf.roll(
            tf.range(len(concat_values.shape)), shift=-1, axis=0)

        concat_values = tf.transpose(concat_values, transpose_indices)

        summed_values = tf.math.unsorted_segment_sum(
            data=concat_values,
            segment_ids=remapped_indices,
            num_segments=tf.reduce_max(remapped_indices) + 1
        )

        summed_values = tf.transpose(summed_values, untranspose_indices)

        # blade_values: [20, 21+24, 22+23, 25]
        # blade_indices: [1, 2, 3, 10]
        return self.with_changes(
            blade_values=summed_values,
            blade_indices=new_indices
        )

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self.with_changes(
            blade_values=-self.blade_values
        )

    def __sub__(self, other):
        neg_other = -self.algebra.as_mv(other)
        return self + neg_other

    def __rsub__(self, other):
        return -self + other

    def __abs__(self):
        return self.with_changes(
            blade_values=tf.abs(self.blade_values)
        )

    def __getitem__(self, key):
        def is_sequence(x):
            return (
                (not hasattr(x, "strip")) and
                hasattr(x, "__getitem__") and
                hasattr(x, "__iter__")
            )

        def is_nonempty_sequence(x):
            return is_sequence(x) and len(x) > 0

        def is_nonempty_str_sequence(x):
            return is_nonempty_sequence(x) and all(isinstance(s, str) for s in x)

        mv = self

        # Find blade index of passed blade strings in global blade index if
        # a string or a sequence of strings was passed in some way.
        # Also remove that part from the key that will later be used for
        # indexing the batch dimensions.
        # Could do this better with a recursive function that generalizes more too.
        blade_indices = None
        if isinstance(key, str):
            # Key is a single string
            blade_indices = tf.convert_to_tensor(
                [[self.algebra.blades.index(key)]], dtype=tf.int64)
            key = tuple()
        elif is_nonempty_str_sequence(key):
            # Key is sequence of strings
            blade_indices = tf.convert_to_tensor(
                [[self.algebra.blades.index(s) for s in key]], dtype=tf.int64)
            key = tuple()
        elif is_nonempty_sequence(key) and isinstance(key[-1], str):
            # Single string at last index of key sequence
            blade_indices = tf.convert_to_tensor(
                [[self.algebra.blades.index(key[-1])]], dtype=tf.int64)
            key = key[:-1]
        elif is_nonempty_sequence(key) and is_nonempty_str_sequence(key[-1]):
            # Sequence of strings at last index of key sequence
            blade_indices = tf.convert_to_tensor(
                [[self.algebra.blades.index(s) for s in key[-1]]], dtype=tf.int64)
            key = key[:-1]

        # Index multi-vector bases if we found an index for them above.
        if blade_indices is not None:
            # Don't allow duplicate indices
            if len(blade_indices[0]) != len(tf.unique(blade_indices[0])[0]):
                raise Exception(
                    "Duplicate blade indices passed: %s" % blade_indices[0])

            # Tile our own sparse blade index across the search indices
            self_blade_indices = tf.tile(tf.expand_dims(
                self.blade_indices, axis=-1), [1, len(blade_indices)])

            # Find the indices of the search indices in our own index
            found_indices = tf.where(
                self_blade_indices == blade_indices)[..., 0]

            if len(found_indices) < blade_indices.shape[1]:
                raise Exception("Could not find all passed blades. Passed blade indices: %s, available blade indices: %s" % (
                    blade_indices[0], self.blade_indices))

            # Get new values and indices at the now known own indices
            new_indices = tf.gather(self.blade_indices, found_indices, axis=-1)
            new_values = tf.gather(self.blade_values, found_indices, axis=-1)

            mv = self.with_changes(
                blade_values=new_values,
                blade_indices=new_indices
            )

        # Use rest of key for normal indexing (usually used for batch indexing).
        # TODO: Need to change blade_indices if they key slices the indices.
        # Could just count the length of the key and look at batch_shape,
        # but that still misses ellipsis.
        return mv.with_changes(
            blade_values=mv.blade_values[key]
        )

    def __setitem__(self, key, value):
        print(key, value)

    def __repr__(self):
        if len(self.blade_values.shape) == 1:
            def _blade_value_repr(value, index):
                blade_repr = get_blade_repr(self.algebra.blades[index])
                return "%.2f*%s" % (value, blade_repr)

            return "MultiVector[%s]" % " + ".join(
                _blade_value_repr(value, index)
                for value, index
                in zip(self.blade_values, self.blade_indices)
                if value != 0
            )
        else:
            return "MultiVector[batch_shape=%s, blades=[%s]]" % (self.batch_shape, ", ".join(get_blade_repr(self.algebra.blades[i]) for i in self.blade_indices))

    def tile(self, multiples):
        expanded_shape = [1] * len(multiples) + [*self.blade_values.shape]
        expanded_multiples = [*multiples] + [1] * len(self.blade_values.shape)

        new_blade_values = tf.tile(
            tf.reshape(self.blade_values, expanded_shape),
            multiples=expanded_multiples
        )

        return self.with_changes(
            blade_values=new_blade_values
        )

    def approx_exp(self, order):
        """Returns an approximation of the exponential using a centered taylor series."""
        v = self.algebra.as_mv(1.0)
        result = self.algebra.as_mv(1.0)
        for i in range(1, order + 1):
            v = self * v
            i_factorial = tf.exp(tf.math.lgamma(i + 1.0))
            result += v / i_factorial
        return result

    def approx_pow(self, x, order):
        """Returns an approximation of the multi-vector to the power of x using a centered taylor series."""
        from scipy.special import binom

        # TODO: binom only works with positive x.
        # Use function that supports negative x.
        b = binom(x, 0, dtype=np.float32)
        result = self.algebra.as_mv(b)

        self_minus_one = self - 1.0
        v = None

        for i in range(1, order + 1):
            v = self_minus_one if v is None else v * self_minus_one
            b = binom(x, i, dtype=np.float32)
            result += b * v

        return result

    def approx_sqrt(self, order):
        """Returns an approximation of the multi-vector to the power of 0.5 using a centered taylor series."""
        return self.approx_pow(0.5, order=order)

    def approx_log(self, order):
        """Returns an approximation of the natural logarithm using a centered taylor series."""
        result = self.algebra.as_mv(0.0)

        self_minus_one = self - 1.0
        v = None

        for i in range(1, order + 1):
            v = self_minus_one if v is None else v * self_minus_one
            result += (((-1.0) ** i) / i) * v

        return -result

    def int_pow(self, n):
        if not isinstance(n, int):
            raise Exception("n must be an integer.")
        if n < 0:
            raise Exception("Can't raise to negative powers.")

        if n == 0:
            return self.algebra.as_mv(1.0)

        result = self
        for i in range(n - 1):
            result *= self
        return result
