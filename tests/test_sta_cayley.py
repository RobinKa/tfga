import tensorflow as tf

# Make tensorflow not take over the entire GPU memory
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

import unittest as ut
import numpy as np

from tfga.cayley import blades_from_bases, get_cayley_tensor

sta_metric = [1, -1, -1, -1]
sta_bases = ["0", "1", "2", "3"]
sta_blades = [
    "",
    "0", "1", "2", "3",
    "01", "02", "03", "12", "13", "23",
    "012", "013", "023", "123",
    "0123"
]
sta_blade_degrees = [len(blade) for blade in sta_blades]


class TestStaCayleyTensor(ut.TestCase):
    def test_sta_blades_from_bases(self):
        blades, blade_degrees = blades_from_bases(sta_bases)
        self.assertCountEqual(blades, sta_blades)
        for blade, blade_degree in zip(sta_blades, sta_blade_degrees):
            blade_index = blades.index(blade)
            self.assertEqual(blade_degrees[blade_index], blade_degree)

    def test_sta_cayley_tensor_scalar_scalar_scalar(self):
        cayley, cayley_inner, cayley_outer = get_cayley_tensor(sta_metric, sta_bases, sta_blades)

        # Scalar * Scalar -> Scalar
        self.assertEqual(cayley[0, 0, 0], 1)
        self.assertTrue(np.all(cayley[0, 0, 1:] == 0))

    def test_sta_cayley_tensor_e12_e23_me13(self):
        cayley, cayley_inner, cayley_outer = get_cayley_tensor(sta_metric, sta_bases, sta_blades)

        # e12 * e23 -> -e13
        e12_index = sta_blades.index("12")
        e23_index = sta_blades.index("23")
        e13_index = sta_blades.index("13")
        self.assertEqual(cayley[e12_index, e23_index, e13_index], -1)
        self.assertTrue(np.all(cayley[e12_index, e23_index, :e13_index] == 0))
        self.assertTrue(
            np.all(cayley[e12_index, e23_index, e13_index+1:] == 0))

    def test_sta_cayley_tensor_e01_e23_e0123(self):
        cayley, cayley_inner, cayley_outer = get_cayley_tensor(sta_metric, sta_bases, sta_blades)

        # e01 * e23 -> e0123
        e01_index = sta_blades.index("01")
        e23_index = sta_blades.index("23")
        e0123_index = sta_blades.index("0123")
        self.assertEqual(cayley[e01_index, e23_index, e0123_index], 1)
        self.assertTrue(
            np.all(cayley[e01_index, e23_index, :e0123_index] == 0))
        self.assertTrue(
            np.all(cayley[e01_index, e23_index, e0123_index+1:] == 0))
