import numpy as np
import pandas as pd

from stochastic_gradient_descent.StochasticGradientDescent import StochasticGradientDescent

if __name__ == '__main__':
    _file_name = 'data_points.csv'

    # load data points
    data_points: np.array = np.loadtxt(_file_name, delimiter=',').T

    sgd = StochasticGradientDescent(data_points, d=10, a=.1, iterations_count=1000)

    print('\033[92m' + 'training...' + '\033[0m')
    sgd.run()

    print('\033[92m' + 'saving data... ' + '\033[0m')
    pd.DataFrame(sgd.history).to_csv('history.csv', index=False)
    print('\033[92m' + 'saved' + '\033[0m')

    # print_repo
    print('REPORT')
    print(f'Alpha: {sgd.a}')
    print(f'Iterations: {sgd.iterations_count}')
    print(f'Polynomial Degree: {sgd.d}')
    print('Initial Parameters:')
    print(sgd.history['polynomial'][0])
    print('Final Parameters:')
    print(sgd.history['polynomial'][-1])
    print('Final Error:')
    print(sgd.history['error'][-1])
