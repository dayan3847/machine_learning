import unittest

from dayan3847.models.multivariate.MultivariateGaussianModel import ModelGaussianMultivariate


class TestMultivariateModel(unittest.TestCase):

    def test_gaussian_2d(self):
        model = ModelGaussianMultivariate(
            a=.1,
            factors_x_dim=[5, 5],
            limits_x_dim=[(0, 1), (0, 1)],
            _s2=.1,
        )

        # self.assertEqual(5, model.f)
        self.assertEqual((1, 25), model.ww.shape)
        self.assertEqual(25, len(model.mm))
        self.assertEqual((2, 1), model.mm[0].shape)
        self.assertEqual((25, 1), model.bb([[1], [1]]).shape)

        g = model.gi([1, 1])

        model.update_w([1, 1], 8)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
