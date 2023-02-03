from src.StochasticGradientDescent import StochasticGradientDescent


class StochasticGradientDescentLineal(StochasticGradientDescent):
    def __init__(self):
        super().__init__()
        self.d = 1
        # Alpha
        self.a: float = .001
        self.iterations_count: int = 1000

    def source_function(self, x):
        return 5 + 10 * x

    # def noise(self):
    #     return 0


if __name__ == '__main__':
    stochastic_gradient_descent = StochasticGradientDescentLineal()
    stochastic_gradient_descent.main()
