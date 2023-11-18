import unittest

from dayan3847.models.linear.LinearPolynomialModel import LinearPolynomialModel
from dayan3847.models.linear.LinearGaussianModel import LinearGaussianModel
from dayan3847.models.linear.LinearSigmoidalModel import LinearSigmoidalModel


class TestLinearModel(unittest.TestCase):

    def test_polynomial(self):
        model = LinearPolynomialModel(5)
        self.assertEqual(5, model.f)
        self.assertEqual((5,), model.ww.shape)
        self.assertEqual((5,), model.nn.shape)
        self.assertEqual((5,), model.bb(2).shape)

        g = model.g_single(2)

        model.train_single(5, 8, a=.1)

    def test_gaussian(self):
        model = LinearGaussianModel(
            (0, 1, 5),
            .1
        )
        self.assertEqual(5, model.f)
        self.assertEqual((5,), model.ww.shape)
        self.assertEqual((5,), model.mm.shape)
        self.assertEqual((5,), model.bb(2).shape)

        g = model.g(2)

        model.train(5, 8, a=.1)

    def test_sigmoidal(self):
        model = LinearSigmoidalModel(
            (0, 1, 5),
            .1
        )
        self.assertEqual(5, model.f)
        self.assertEqual((5,), model.ww.shape)
        self.assertEqual((5,), model.mm.shape)
        self.assertEqual((5,), model.bb(2).shape)

        g = model.g(2)

        model.train(5, 8, a=.1)


if __name__ == '__main__':
    unittest.main()
