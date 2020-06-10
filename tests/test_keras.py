import unittest as ut
import tensorflow as tf
from tensorflow import keras as ks

from tfga import GeometricAlgebra
from tfga.blades import BladeKind
from tfga.layers import (
    GeometricProductDense,
    GeometricToTensor, GeometricToTensorWithKind,
    TensorToGeometric, TensorWithKindToGeometric
)


class TestKerasLayers(ut.TestCase):
    def assertTensorsEqual(self, a, b):
        self.assertTrue(tf.reduce_all(a == b), "%s not equal to %s" % (a, b))

    def test_tensor_to_geometric(self):
        sta = GeometricAlgebra([1, -1, -1, -1])
        tensor = tf.ones([32, 4])
        gt_geom_tensor = tf.concat(
            [tf.zeros([32, 1]), tf.ones([32, 4]), tf.zeros([32, 11])],
            axis=-1
        )

        vector_blade_indices = [1, 2, 3, 4]

        tensor_to_geom_layer = TensorToGeometric(sta, vector_blade_indices)

        self.assertTensorsEqual(tensor_to_geom_layer(tensor), gt_geom_tensor)

    def test_tensor_with_kind_to_geometric(self):
        sta = GeometricAlgebra([1, -1, -1, -1])
        tensor = tf.ones([32, 4])
        gt_geom_tensor = tf.concat(
            [tf.zeros([32, 1]), tf.ones([32, 4]), tf.zeros([32, 11])],
            axis=-1
        )

        vector_blade_indices = [1, 2, 3, 4]

        tensor_kind_to_geom_layer = TensorWithKindToGeometric(
            sta, BladeKind.VECTOR)

        self.assertTensorsEqual(
            tensor_kind_to_geom_layer(tensor), gt_geom_tensor)

    def test_geometric_to_tensor(self):
        sta = GeometricAlgebra([1, -1, -1, -1])
        gt_tensor = tf.ones([32, 4])
        geom_tensor = tf.concat(
            [tf.zeros([32, 1]), tf.ones([32, 4]), tf.zeros([32, 11])],
            axis=-1
        )

        vector_blade_indices = [1, 2, 3, 4]

        geom_to_tensor_layer = GeometricToTensor(sta, vector_blade_indices)

        self.assertTensorsEqual(geom_to_tensor_layer(geom_tensor), gt_tensor)

    def test_geometric_to_tensor_with_kind(self):
        sta = GeometricAlgebra([1, -1, -1, -1])
        gt_tensor = tf.ones([32, 4])
        geom_tensor = tf.concat(
            [tf.zeros([32, 1]), tf.ones([32, 4]), tf.zeros([32, 11])],
            axis=-1
        )

        vector_blade_indices = [1, 2, 3, 4]

        geom_to_tensor_kind_layer = GeometricToTensorWithKind(
            sta, BladeKind.VECTOR)

        self.assertTensorsEqual(
            geom_to_tensor_kind_layer(geom_tensor), gt_tensor)

    def test_geometric_product_dense_mv(self):
        sta = GeometricAlgebra([1, -1, -1, -1])

        geom_tensor = tf.concat(
            [tf.zeros([32, 6, 1]), tf.ones([32, 6, 4]), tf.zeros([32, 6, 11])],
            axis=-1
        )

        geom_prod_layer = GeometricProductDense(
            sta, 8,
            blade_indices_w=[1, 2, 3, 4], blade_indices_b=[0]
        )

        result = geom_prod_layer(geom_tensor)
