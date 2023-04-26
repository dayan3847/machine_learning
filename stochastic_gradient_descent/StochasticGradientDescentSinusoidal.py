from stochastic_gradient_descent.StochasticGradientDescent import StochasticGradientDescent
import numpy as np


class StochasticGradientDescentSinusoidal(StochasticGradientDescent):

    def get_x_list(self):
        return self.get_x_list_random()

    def source_function(self, x):
        return np.sin(2 * np.pi * x)

    # def noise(self):
    #     return 0


if __name__ == '__main__':
    stochastic_gradient_descent = StochasticGradientDescentSinusoidal()
    # Config
    stochastic_gradient_descent.a = .1
    stochastic_gradient_descent.d = 10
    stochastic_gradient_descent.iterations_count = 5000
    # Run
    stochastic_gradient_descent.main()
