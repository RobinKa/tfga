"""Operations on multivectors used internally."""
import tensorflow as tf


def mv_multiply(a_blade_indices: tf.Tensor, a_blade_values: tf.Tensor,
                b_blade_indices: tf.Tensor, b_blade_values: tf.Tensor,
                cayley: tf.Tensor) -> tf.Tensor:
    sub_cayley = tf.gather(cayley, a_blade_indices, axis=0)
    sub_cayley = tf.gather(sub_cayley, b_blade_indices, axis=1)

    # [N, 3]
    x = tf.where(sub_cayley != 0)[:, 2]
    result_blade_indices, _ = tf.unique(x, tf.int64)
    sub_cayley = tf.gather(sub_cayley, result_blade_indices, axis=2)

    result_blade_values = tf.einsum("...i,...j,...ijk->...k",
                                    a_blade_values,
                                    b_blade_values,
                                    sub_cayley)

    return result_blade_indices, result_blade_values


def mv_add(a_blade_indices: tf.Tensor, a_blade_values: tf.Tensor,
           b_blade_indices: tf.Tensor, b_blade_values: tf.Tensor) -> tf.Tensor:
    # vals: [20, 21, 22] [23, 24, 25]
    # ind: [1, 2, 3], [3, 2, 10]

    # concat indices: [1, 2, 3, 3, 2, 10]
    concat_indices = tf.concat(
        [a_blade_indices, b_blade_indices],
        axis=0
    )

    # new_indices: [1, 2, 3, 10]
    # remapped_indices (index of old values in new_indices): [0, 1, 2, 2, 1, 3]
    new_indices, remapped_indices = tf.unique(concat_indices, tf.int64)

    # concat values: [20, 21, 22, 23, 24, 25]
    concat_values = tf.concat(
        [a_blade_values, b_blade_values],
        # axis=-1 but can't use negative indexing because that creates new axes
        axis=len(a_blade_values.shape) - 1
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

    # blade_indices: [1, 2, 3, 10]
    # blade_values: [20, 21+24, 22+23, 25]
    return new_indices, summed_values


def mv_equal(a_blade_indices: tf.Tensor, a_blade_values: tf.Tensor,
             b_blade_indices: tf.Tensor, b_blade_values: tf.Tensor) -> tf.Tensor:
    # TODO: Make sure blade indices are broadcastable.

    # Align a and b indices (if possible)
    # to the compare all indices and values.
    reindex_a = tf.argsort(a_blade_indices)
    reindex_b = tf.argsort(b_blade_indices)

    sorted_ind_a = tf.gather(a_blade_indices, reindex_a, axis=-1)
    sorted_ind_b = tf.gather(b_blade_indices, reindex_b, axis=-1)

    sorted_val_a = tf.gather(a_blade_values, reindex_a, axis=-1)
    sorted_val_b = tf.gather(b_blade_values, reindex_b, axis=-1)

    return tf.reduce_all(sorted_ind_a == sorted_ind_b) and tf.reduce_all(sorted_val_a == sorted_val_b)


def mv_reversion(a_blade_indices, a_blade_values, algebra_blade_degrees):
    blade_degrees = tf.cast(
        tf.gather(algebra_blade_degrees, a_blade_indices), tf.float32)

    # for each blade, 0 if even number of swaps required, else 1
    odd_swaps = tf.cast(
        tf.floor(blade_degrees * (blade_degrees - 0.5)) % 2, tf.float32)

    # [0, 1] -> [-1, 1]
    reversion_signs = 1.0 - 2.0 * odd_swaps

    return reversion_signs * a_blade_values


def mv_grade_automorphism(a_blade_indices, a_blade_values, algebra_blade_degrees):
    blade_degrees = tf.cast(
        tf.gather(algebra_blade_degrees, a_blade_indices), tf.float32)
    signs = 1.0 - 2.0 * (blade_degrees % 2)
    return signs * a_blade_values
