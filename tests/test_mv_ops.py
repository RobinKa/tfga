import unittest as ut
import tensorflow as tf

from tfga.mv_ops import mv_add, mv_multiply


class TestMultiVectorOps(ut.TestCase):
    def test_mv_add(self):
        a_blade_indices = tf.constant([
            0, 1, 2
        ], dtype=tf.int64)

        a_blade_values = tf.constant([
            11, 12, 13
        ], dtype=tf.float32)

        b_blade_indices = tf.constant([
            2, 1, 4
        ], dtype=tf.int64)

        b_blade_values = tf.constant([
            14, 16, 20
        ], dtype=tf.float32)

        desired_results = {
            0: 11,
            1: 12 + 16,
            2: 13 + 14,
            4: 20
        }

        result_blade_indices, result_blade_values = mv_add(
            a_blade_indices, a_blade_values,
            b_blade_indices, b_blade_values
        )

        unique_indices = tf.constant([0, 1, 2, 4], dtype=tf.int64)

        self.assertEqual(
            set(result_blade_indices.numpy()),
            set(unique_indices.numpy())
        )

        for result_blade_index, result_blade_value in zip(result_blade_indices, result_blade_values):
            self.assertEqual(result_blade_value, desired_results[result_blade_index.numpy()])
