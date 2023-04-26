from abc import ABC, abstractmethod
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


class StochasticGradientDescent(ABC):

    def __init__(self):
        self.last_plot_time = -1
        self.plot_interval = 2
        self.save_best_parameters = False
        self.sleep_for_each_iteration = 0
        self.graphics = None
        self.min_error = None
        # Artificial Count
        self.m: int = 100
        # Polynomial Degree
        self.d: int = 10
        # Alpha
        self.a: float = .001
        # Iterations Count
        self.iterations_count: int = 1000

        self.x_interval = (0, 1)
        self.epsilon_interval = (-.3, .3)
        self.initial_parameters_interval = (-.5, .5)

        self.errors = []

        self.data_point = None
        # Parameters of the polynomial
        self.parameters = None
        self.initial_parameters = None

        self.best_parameters = None

    def init(self):
        self.generate_data_points()
        self.generate_initial_parameters()
        if self.save_best_parameters:
            self.best_parameters = self.initial_parameters.copy()
        self.min_error = self.error()

    # Generate Data Points
    def generate_data_points(self):
        if self.data_point is not None:
            return
        self.data_point = {
            'x_list': self.get_x_list(),
            'y_list': [],
        }
        for x in self.data_point['x_list']:
            e = self.noise()
            y = self.source_function(x) + e
            self.data_point['y_list'].append(y)

    @abstractmethod
    def source_function(self, x):
        pass

    def noise(self):
        return random.uniform(self.epsilon_interval[0], self.epsilon_interval[1])

    def get_x_list(self):
        return self.get_x_list_ordered()

    def get_x_list_ordered(self):
        step = (self.x_interval[1] - self.x_interval[0]) / self.m
        return np.arange(self.x_interval[0], self.x_interval[1], step)

    def get_x_list_random(self):
        x_list = []
        for _ in range(self.m):
            x = random.uniform(self.x_interval[0], self.x_interval[1])
            x_list.append(x)
        return x_list

    # Generate Parameters
    def generate_initial_parameters(self):
        if self.graphics is None:
            self.graphics = {
                'x': np.arange(0, 1, 0.01),
                'y': [],
                'y_init': [],
                'y_best': [],
            }
        if self.initial_parameters is not None:
            return
        self.initial_parameters = []
        for i in range(self.d + 1):
            parameter = random.uniform(self.initial_parameters_interval[0], self.initial_parameters_interval[1])
            self.initial_parameters.append(parameter)

        self.parameters = self.initial_parameters.copy()
        y_list = self.get_current_y_list()
        self.graphics['y'] = y_list
        self.graphics['y_init'] = y_list
        self.graphics['y_best'] = y_list

    def get_current_y_list(self):
        y_list = []
        for x in self.graphics['x']:
            y = self.h(x)
            y_list.append(y)
        return y_list

    def update_graphics(self, graphic: str = 'y'):
        self.graphics[graphic] = self.get_current_y_list()

    def h(self, x):
        result = 0
        for i in range(self.d + 1):
            result += self.parameters[i] * (x ** i)
        return result

    def error(self):
        error = 0
        for i in range(self.m):
            xi = self.data_point['x_list'][i]
            yi = self.data_point['y_list'][i]
            hi = self.h(xi)
            error += (hi - yi) ** 2
        return error / 2

    def error_rms(self):
        e = self.error()
        return (2 * e / self.m) ** 0.5

    def run(self, plot: bool = False):
        for k in range(self.iterations_count):
            for i in range(self.m):
                xi = self.data_point['x_list'][i]
                yi = self.data_point['y_list'][i]
                for j in range(self.d + 1):
                    hi = self.h(xi)
                    self.parameters[j] += self.a * (yi - hi) * (xi ** j)
            ek = self.error_rms()
            if self.save_best_parameters and ek < self.min_error:
                self.min_error = ek
                self.best_parameters = self.parameters.copy()
                self.update_graphics('y_best')
            self.errors.append(ek)
            if plot:
                clear_output(wait=True)
                self.update_graphics()
                self.plot_data_all()
            time.sleep(self.sleep_for_each_iteration)

    def plot_data(
            self,
            initial: bool = False,
            best: bool = False,
            current: bool = False,
    ):
        # Data
        plt.clf()
        plt.title('Data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axvline(color='black')
        plt.axhline(color='black')
        # points
        dpx = self.data_point['x_list']
        dpy = self.data_point['y_list']
        plt.scatter(dpx, dpy, color='gray', label='data points')
        if initial or best or current:
            x = self.graphics['x']
            # initial
            if initial:
                y_init = self.graphics['y_init']
                plt.plot(x, y_init, color='blue', label='initial')
            if best and self.save_best_parameters:
                y_best = self.graphics['y_best']
                plt.plot(x, y_best, color='yellow', label='best')
            if current:
                y = self.graphics['y']
                plt.plot(x, y, color='green', label='current')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_data_all(self, force: bool = True):
        current_time = int(time.time())
        if not force and current_time - self.last_plot_time < self.plot_interval:
            return
        self.last_plot_time = current_time
        # Data
        self.plot_data(True, True, True)
        # Error
        self.plot_error()

    def plot_error(self):
        plt.clf()
        plt.title('Error')
        plt.xlabel('i')
        plt.ylabel('error')
        plt.plot(self.errors, color='red', label='error')
        plt.legend()
        plt.grid()
        plt.show()

    def print_repo(self):
        print('REPORT')
        print(f'Alpha: {self.a}')
        print(f'Iterations: {self.iterations_count}')
        print(f'Polynomial Degree: {self.d}')
        print('Initial Parameters:')
        print(self.initial_parameters)
        print('Final Parameters:')
        print(self.parameters)
        print('Final Error:')
        print(self.errors[-1])

    def main(self, plot: bool = False):
        self.init()
        self.run(plot)
        if plot:
            clear_output(wait=True)
        else:
            self.update_graphics()
        self.plot_data_all()
        self.print_repo()
