from stochastic_gradient_descent.StochasticGradientDescent import StochasticGradientDescent


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
    # Config
    stochastic_gradient_descent.a = .001
    stochastic_gradient_descent.iterations_count = 100
    # Run
    stochastic_gradient_descent.main()
