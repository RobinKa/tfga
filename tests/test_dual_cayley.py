import unittest as ut
from tfga.cayley import blades_from_bases, get_cayley_tensor

dual_metric = [0]
dual_bases = ["0"]
dual_blades = ["", "0"]
dual_blade_degrees = [len(blade) for blade in dual_blades]


class TestDualCayleyTensor(ut.TestCase):
    def test_dual_blades_from_bases(self):
        blades, blade_degrees = blades_from_bases(dual_bases)
        self.assertCountEqual(blades, dual_blades)
        for blade, blade_degree in zip(dual_blades, dual_blade_degrees):
            blade_index = blades.index(blade)
            self.assertEqual(blade_degrees[blade_index], blade_degree)

    def test_cayley_tensor_correct(self):
        cayley, cayley_inner, cayley_outer = get_cayley_tensor(
            dual_metric, dual_bases, dual_blades)

        self.assertSequenceEqual(cayley.shape, [2, 2, 2])

        # Scalar * Scalar -> Scalar
        self.assertEqual(cayley[0, 0, 0], 1)
        self.assertEqual(cayley[0, 0, 1], 0)

        # Scalar * Dual -> Dual
        self.assertEqual(cayley[0, 1, 0], 0)
        self.assertEqual(cayley[0, 1, 1], 1)

        # Dual * Scalar -> Dual
        self.assertEqual(cayley[1, 0, 0], 0)
        self.assertEqual(cayley[1, 0, 1], 1)

        # Dual * Dual -> Zero
        self.assertEqual(cayley[1, 1, 0], 0)
        self.assertEqual(cayley[1, 1, 1], 0)
