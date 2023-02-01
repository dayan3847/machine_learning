from src.StochasticGradientDescent import StochasticGradientDescent


class StochasticGradientDescentCubic(StochasticGradientDescent):
    def __init__(self):
        super().__init__()
        self.d = 3
        # Alpha
        self.a: float = .001
        self.iterations_count: int = 1000

    def source_function(self, x):
        return 5 + 0 * x - 9 * x ** 2 + 10 * x ** 3

    # def noise(self):
    #     return 0


if __name__ == '__main__':
    stochastic_gradient_descent = StochasticGradientDescentCubic()
    stochastic_gradient_descent.main()
