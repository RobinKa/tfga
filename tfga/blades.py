"""Blade-related definitions and functions used across the library."""
from enum import Enum
import tensorflow as tf


class BladeKind(Enum):
    """Kind of blade depending on its degree."""
    MV = "mv"
    EVEN = "even"
    ODD = "odd"
    SCALAR = "scalar"
    PSEUDOSCALAR = "pseudoscalar"
    VECTOR = "vector"
    PSEUDOVECTOR = "pseudovector"


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
    elif kind == BladeKind.PSEUDOSCALAR.value:
        return blade_degrees == max_degree
    elif kind == BladeKind.VECTOR.value:
        return blade_degrees == 1
    elif kind == BladeKind.PSEUDOVECTOR.value:
        return blade_degrees == max_degree - 1
    raise Exception("Unknown blade kind: %s" % kind)


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
