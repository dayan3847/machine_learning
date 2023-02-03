import math

from src.StochasticGradientDescent import StochasticGradientDescent


class StochasticGradientDescentSinusoidal(StochasticGradientDescent):
    def source_function(self, x):
        return math.sin(2 * math.pi * x)

    # def noise(self):
    #     return 0


if __name__ == '__main__':
    stochastic_gradient_descent = StochasticGradientDescentSinusoidal()
    # Config
    stochastic_gradient_descent.a = .1
    stochastic_gradient_descent.d = 10
    stochastic_gradient_descent.iterations_count = 50
    # Run
    stochastic_gradient_descent.main()
