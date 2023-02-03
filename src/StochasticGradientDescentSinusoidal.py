import math

from src.StochasticGradientDescent import StochasticGradientDescent


class StochasticGradientDescentSinusoidal(StochasticGradientDescent):
    def __init__(self):
        super().__init__()
        self.a: float = 0.4
        self.d = 5
        self.iterations_count: int = 5000

    def source_function(self, x):
        return math.sin(2 * math.pi * x)

    # def noise(self):
    #     return 0


if __name__ == '__main__':
    stochastic_gradient_descent = StochasticGradientDescentSinusoidal()
    stochastic_gradient_descent.main()
