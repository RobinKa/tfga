"""Blade-related definitions and functions used across the library."""
from enum import Enum
from typing import List, Tuple
import tensorflow as tf


class BladeKind(Enum):
    """Kind of blade depending on its degree."""
    MV = "mv"
    EVEN = "even"
    ODD = "odd"
    SCALAR = "scalar"
    VECTOR = "vector"
    BIVECTOR = "bivector"
    TRIVECTOR = "trivector"
    PSEUDOSCALAR = "pseudoscalar"
    PSEUDOVECTOR = "pseudovector"
    PSEUDOBIVECTOR = "pseudobivector"
    PSEUDOTRIVECTOR = "pseudotrivector"


def get_blade_repr(blade_name: str) -> str:
    """Returns the representation to use
    for a given blade.

    Examples:
    - `"12"` -> `"e_12"`
    - `""` -> `"1"`

    Args:
        blade_name: name of the blade in the algebra (eg. `"12"`)

    Returns:
        Representation to use for a given blade
    """
    if blade_name == "":
        return "1"
    return "e_%s" % blade_name


def is_blade_kind(blade_degrees: tf.Tensor, kind: [BladeKind, str], max_degree: int) -> tf.Tensor:
    """Finds a boolean mask for whether blade degrees are of a given kind.

    Args:
        blade_degrees: list of blade degrees
        kind: kind of blade to check for
        max_degree: maximum blade degree in the algebra

    Returns:
        boolean mask for whether blade degrees are of a given kind
    """
    # Convert kind to string representation
    # for comparison.
    kind = kind.value if isinstance(kind, BladeKind) else kind

    if kind == BladeKind.MV.value:
        return tf.constant(True, shape=[len(blade_degrees)])
    elif kind == BladeKind.EVEN.value:
        return blade_degrees % 2 == 0
    elif kind == BladeKind.ODD.value:
        return blade_degrees % 2 == 1
    elif kind == BladeKind.SCALAR.value:
        return blade_degrees == 0
    elif kind == BladeKind.VECTOR.value:
        return blade_degrees == 1
    elif kind == BladeKind.BIVECTOR.value:
        return blade_degrees == 2
    elif kind == BladeKind.TRIVECTOR.value:
        return blade_degrees == 3
    elif kind == BladeKind.PSEUDOSCALAR.value:
        return blade_degrees == max_degree
    elif kind == BladeKind.PSEUDOVECTOR.value:
        return blade_degrees == max_degree - 1
    elif kind == BladeKind.PSEUDOBIVECTOR.value:
        return blade_degrees == max_degree - 2
    elif kind == BladeKind.PSEUDOTRIVECTOR.value:
        return blade_degrees == max_degree - 3
    raise Exception("Unknown blade kind: %s" % kind)


def invert_blade_indices(num_blades: int, blade_indices: tf.Tensor) -> tf.Tensor:
    """Returns all blade indices except for the given ones.

    Args:
        num_blades: Total number of blades in the algebra
        blade_indices: blade indices to exclude

    Returns:
        All blade indices except for the given ones
    """

    all_blades = tf.range(num_blades, dtype=blade_indices.dtype)
    return tf.sparse.to_dense(tf.sets.difference(
        tf.expand_dims(all_blades, axis=0),
        tf.expand_dims(blade_indices, axis=0)
    ))[0]


def get_blade_of_kind_indices(blade_degrees: tf.Tensor, kind: BladeKind,
                              max_degree: int, invert: bool = False) -> tf.Tensor:
    """Finds a boolean mask for whether blades are of a given kind.

    Args:
        blade_degrees: List of blade degrees
        kind: kind of blade for which the mask will be true
        max_degree: maximum blade degree in the algebra
        invert: whether to invert the result

    Returns:
        boolean mask for whether blades are of a given kind
    """
    cond = is_blade_kind(blade_degrees, kind, max_degree)
    cond = tf.math.logical_xor(cond, invert)
    return tf.where(cond)[:, 0]


def _normal_swap(x: List[str]) -> List[str]:
    """Swaps the first unordered blade pair and returns the new list as well
    as whether a swap was performed."""
    for i in range(len(x) - 1):
        a, b = x[i], x[i + 1]
        if a > b:  # string comparison
            x[i], x[i+1] = b, a
            return False, x
    return True, x


def get_normal_ordered(blade_name: str) -> Tuple[int, str]:
    """Returns the normal ordered blade name and its sign.
    Example: 21 => -1, 12

    Args:
        blade_name: Blade name for which to return normal ordered
        name and sign

    Returns:
        sign: sign of the blade
        blade_name: normalized name of the blade
    """
    blade_name = list(blade_name)
    sign = -1
    done = False
    while not done:
        sign *= -1
        done, blade_name = _normal_swap(blade_name)
    return sign, "".join(blade_name)


def get_blade_indices_from_names(blade_names: List[str],
                                 all_blade_names: List[str]) -> tf.Tensor:
    """Finds blade signs and indices for given blade names in a list of blade
    names. Blade names can be unnormalized and their correct sign will be
    returned.

    Args:
        blade_names: Blade names to return indices for. May be unnormalized.
        all_blade_names: Blade names to use as index

    Returns:
        blade_signs: signs for the passed blades in same order as passed
        blade_indices: blade indices in the same order as passed
    """
    signs_and_names = [get_normal_ordered(b) for b in blade_names]

    blade_signs = [sign for sign, blade_name in signs_and_names]

    blade_indices = [
        all_blade_names.index(blade_name)
        for sign, blade_name in signs_and_names
    ]

    return (tf.convert_to_tensor(blade_signs, dtype=tf.float32),
            tf.convert_to_tensor(blade_indices, dtype=tf.int64))
