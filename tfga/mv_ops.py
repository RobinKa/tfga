"""Operations on geometric algebra tensors used internally."""
import tensorflow as tf


def mv_multiply(a_blade_values: tf.Tensor, b_blade_values: tf.Tensor,
                cayley: tf.Tensor) -> tf.Tensor:
    # ...i, ijk -> ...jk
    x = tf.tensordot(a_blade_values, cayley, axes=[-1, 0])

    # ...1j, ...jk -> ...1k
    x = tf.expand_dims(b_blade_values, axis=b_blade_values.shape.ndims - 1) @ x

    # ...1k -> ...k
    x = tf.squeeze(x, axis=-2)

    return x


def mv_reversion(a_blade_values, algebra_blade_degrees):
    algebra_blade_degrees = tf.cast(algebra_blade_degrees, tf.float32)

    # for each blade, 0 if even number of swaps required, else 1
    odd_swaps = tf.cast(tf.floor(
        algebra_blade_degrees * (algebra_blade_degrees - 0.5)
    ) % 2, tf.float32)

    # [0, 1] -> [-1, 1]
    reversion_signs = 1.0 - 2.0 * odd_swaps

    return reversion_signs * a_blade_values


def mv_grade_automorphism(a_blade_values, algebra_blade_degrees):
    algebra_blade_degrees = tf.cast(algebra_blade_degrees, tf.float32)
    signs = 1.0 - 2.0 * (algebra_blade_degrees % 2.0)
    return signs * a_blade_values
