from src.StochasticGradientDescent import StochasticGradientDescent


class StochasticGradientDescentConstant(StochasticGradientDescent):
    def __init__(self):
        super().__init__()
        self.d = 0

    def source_function(self, x):
        return 10

    def noise(self):
        return 0


if __name__ == '__main__':
    stochastic_gradient_descent = StochasticGradientDescentConstant()
    stochastic_gradient_descent.main()
