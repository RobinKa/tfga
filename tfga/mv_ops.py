"""Operations on geometric algebra tensors used internally."""
from typing import Union
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


def mv_conv1d(a_blade_values: tf.Tensor, k_blade_values: tf.Tensor, cayley: tf.Tensor,
              stride: int, padding: str,
              dilations: Union[int, None] = None) -> tf.Tensor:
    # Winograd convolution

    # A: [..., S, CI, BI]
    # K: [K, CI, CO, BK]
    # C: [BI, BK, BO]

    kernel_size = k_blade_values.shape[0]

    a_batch_shape = tf.shape(a_blade_values)[:-3]

    # Reshape a_blade_values to a 2d image (since that's what the tf op expects)
    # [*, S, 1, CI*BI]
    a_image_shape = tf.concat([
        a_batch_shape,
        tf.shape(a_blade_values)[-3:-2],
        [1, tf.reduce_prod(tf.shape(a_blade_values)[-2:])]
    ], axis=0)
    a_image = tf.reshape(a_blade_values, a_image_shape)

    sizes = [1, kernel_size, 1, 1]
    strides = [1, stride, 1, 1]

    # [*, P, 1, K*CI*BI] where eg. number of patches P = S * K for
    # stride=1 and "SAME", (S-K+1) * K for "VALID", ...
    a_slices = tf.image.extract_patches(
        a_image,
        sizes=sizes, strides=strides,
        rates=[1, 1, 1, 1], padding=padding
    )

    # [..., P, K, CI, BI]
    out_shape = tf.concat([
        a_batch_shape,
        tf.shape(a_slices)[-3:-2],
        tf.shape(k_blade_values)[:1],
        tf.shape(a_blade_values)[-2:]
    ], axis=0)

    a_slices = tf.reshape(a_slices, out_shape)

    # TODO: Optimize this to not use einsum (since it's slow with ellipses)
    # a_...p,k,ci,bi; k_k,ci,co,bk; c_bi,bk,bo -> y_...p,co,bo
    #   ...a b c  d ,   e c  f  g ,   d  g  h  ->   ...a f  h
    x = tf.einsum("...abcd,bcfg,dgh->...afh", a_slices, k_blade_values, cayley)

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
