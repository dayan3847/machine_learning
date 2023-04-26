from stochastic_gradient_descent.StochasticGradientDescent import StochasticGradientDescent


class StochasticGradientDescentQuadratic(StochasticGradientDescent):
    def __init__(self):
        super().__init__()
        self.d = 2

    def source_function(self, x):
        return 5 + 10 * x + 5 * x ** 2

    def noise(self):
        return 0


if __name__ == '__main__':
    stochastic_gradient_descent = StochasticGradientDescentQuadratic()
    # Config
    stochastic_gradient_descent.a = .001
    stochastic_gradient_descent.iterations_count = 100
    # Run
    stochastic_gradient_descent.main()
