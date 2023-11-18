import unittest
import numpy as np

from dayan3847.models.multivariate.MultivariateGaussianModel import MultivariateGaussianModel


class TestMultivariateModel(unittest.TestCase):

    def test_gaussian_2d(self):
        model = MultivariateGaussianModel(
            [
                (0, 1, 5),
                (0, 1, 5),
            ],
            cov=np.array([
                [.1, 0],
                [0, .1],
            ]),
        )

        # self.assertEqual(5, model.f)
        self.assertEqual((25,), model.ww.shape)
        self.assertEqual((25, 2), model.mm.shape)
        self.assertEqual((25,), model.bb([1, 1]).shape)

        g = model.g([1, 1])

        model.train([1, 1], 8, a=.1)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
