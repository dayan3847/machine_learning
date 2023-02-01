from src.StochasticGradientDescent import StochasticGradientDescent


class StochasticGradientDescentLineal(StochasticGradientDescent):
    def __init__(self):
        super().__init__()
        self.d = 1
        # Alpha
        self.a: float = .001
        self.iterations_count: int = 1000

    def source_function(self, x):
        return 10 * x + 5

    def noise(self):
        return 0


if __name__ == '__main__':
    stochastic_gradient_descent = StochasticGradientDescentLineal()
    stochastic_gradient_descent.init()
    stochastic_gradient_descent.run()
    stochastic_gradient_descent.plot_errors()
    stochastic_gradient_descent.plot_data()
