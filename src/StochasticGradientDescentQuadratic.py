from src.StochasticGradientDescent import StochasticGradientDescent


class StochasticGradientDescentQuadratic(StochasticGradientDescent):
    def __init__(self):
        super().__init__()
        self.d = 2
        # Alpha
        self.a: float = .001
        self.iterations_count: int = 1000

    def source_function(self, x):
        return 5 + 10 * x + 5 * x ** 2

    def noise(self):
        return 0


if __name__ == '__main__':
    stochastic_gradient_descent = StochasticGradientDescentQuadratic()
    stochastic_gradient_descent.main()
