import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import threading
import multiprocessing

from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from dayan3847.basis_functions import gaussian_multivariate_2d
from dayan3847.tools import ShapeChecker, PlotterData


class Model2D:
    def __init__(self, factors_x_dim: int, epochs: int, a: float):
        self.a: float = a  # Learning rate
        self.factors_x_dim: int = factors_x_dim  # Number of factors per dimension
        self.epochs: int = epochs  # Number of epochs

        # the data would be in a 3xN matrix
        # where 3 is the dimension of the data and N is the number of data
        # de los 3, el ultimo es de Y y los anteriores forman el vector X
        self.data: np.array = np.loadtxt('data_3d.csv', delimiter=',').T  # Load Data
        ShapeChecker.check_shape_point_set(self.data, 3)
        self.data_x: np.array = self.data[0:2]
        self.data_y: np.array = self.data[2]

        self.n: int = self.data.shape[1]  # Examples count
        # dimensions of x
        self.data_limits: tuple = (0, 1)

        self.f: int = self.factors_x_dim ** 2  # Number of factors
        # Weights: Factors Row Vector Ex. shape(1,25,)
        self.w_vfr: np.array = np.random.rand(self.f)[np.newaxis, :] - .5  # Weights
        # aumentar el peso del medio
        # self.w_vfr[0, self.f // 2] = 5

        self.error_history: np.array = np.array([])  # Error history

        self.thread: threading.Thread = threading.Thread(target=self.train_callback)
        self.queue_error: multiprocessing.Queue = multiprocessing.Queue()
        self.process: multiprocessing.Process = multiprocessing.Process(
            target=self.train_callback,
            args=(self.queue_error,),
        )

        # Check shapes
        ShapeChecker.check_shape(self.w_vfr, (1, self.f,))

    # Calculate the basis function a partir de un valor X
    # retorna un vector de shape(25,1,)
    # cada valor del vector es el resultado de aplicar la funcion base que se multiplica por cada peso
    # entrada esperada es un punto X (2,1) donde 2 es la cantidad de dimensiones de X
    def basis_function__(self, x_: np.array) -> np.array:
        ShapeChecker.check_shape(x_, (2, 1))

    # Calculate the model value for a simple value

    # entrada esperada: Ex: shape(2,) donde 2 es la cantidad de dimensiones de X
    # Expected shape: (x_dim,)
    def hi(self, x_: np.array) -> float:
        # Check shapes
        ShapeChecker.check_shape(x_, (2, 1))
        bfi: np.array = self.basis_function__(x_)
        # Check shapes
        ShapeChecker.check_shape(bfi, (self.f, 1))
        _r = self.w_vfr @ bfi
        return float(_r[0, 0])

    def activate(self, h: float) -> float:
        # return 1 / (1 + np.exp(-h))
        return h

    # Expected shape: (x_dim,)
    def gi(self, x_: np.array) -> float:
        # Check shapes
        ShapeChecker.check_shape(x_, (2, 1))
        return self.activate(self.hi(x_))

    # Expected shape: (x_dim,N )
    def g(self, x_set_: np.array) -> np.array:
        # Check shapes
        ShapeChecker.check_shape_point_set(x_set_, 2)
        _r_set: np.array = np.array([])
        for _x_1d in x_set_.T:
            _x = _x_1d[:, np.newaxis]
            _r_i = self.gi(_x)
            _r_set = np.append(_r_set, _r_i)
        ShapeChecker.check_shape(_r_set, (x_set_.shape[1],))
        return _r_set

    def e(self) -> float:
        _g_set: np.array = self.g(self.data_x)
        _y_set: np.array = self.data_y
        _diff_set: np.array = _g_set - _y_set
        _e: float = np.sum(_diff_set ** 2) / 2
        return _e

    def summary(self):
        print('Model: {}'.format(self.__class__.__name__))
        print('Error: {}'.format(round(self.e(), 2)))

    def train_callback(self, queue_error: multiprocessing.Queue):
        self.queue_error = queue_error
        for _ep in range(self.epochs):
            print('epoch: {} error: {}'.format(_ep, self.e()))
            self.save()
            self.train_step()

    def train(self):
        # self.thread.start()
        self.process.start()

    def train_step(self):
        for _x_1d, _y in zip(self.data_x.T, self.data_y):
            _x = _x_1d[:, np.newaxis]
            _g: float = self.gi(_x)
            _diff: float = _g - _y
            _a_diff: float = self.a * _diff
            _bf_vfr: np.array = self.basis_function__(_x)
            ShapeChecker.check_shape(_bf_vfr, (self.f, 1))
            _dw_vfc: np.array = _a_diff * _bf_vfr
            self.w_vfr -= _dw_vfc.T

    def save(self):
        _error: float = self.e()
        self.error_history = np.append(self.error_history, _error)
        self.queue_error.put(_error)


class Model2DGaussian(Model2D):

    def __init__(self, factors_x_dim: int, epochs: int, a: float, _s2: float):
        super().__init__(factors_x_dim, epochs, a)
        self.mm__: list[np.array] = self.get_mu_list()
        # covariance matrix identity
        self.cov: np.array = np.identity(2) * _s2
        self.cov_inv: np.array = np.linalg.inv(self.cov)
        # Verify shapes
        ShapeChecker.check_shape(self.mm__[0], (2, 1))
        ShapeChecker.check_shape(self.cov, (2, 2))

    # Lista de vectores mu, cada vector mu es un vector columna de shape (2,1)
    def get_mu_list(self) -> list[np.array]:
        _r: list[np.array] = []
        _possible_m: np.array = np.linspace(self.data_limits[0], self.data_limits[1], self.factors_x_dim)
        _x0, _x1 = np.meshgrid(_possible_m, _possible_m)
        _x0 = _x0.flatten()
        _x1 = _x1.flatten()
        for _x0_i, _x1_i in zip(_x0, _x1):
            _r_i = np.array([_x0_i, _x1_i])[:, np.newaxis]
            _r.append(_r_i)

        return _r

    def basis_function__(self, x_: np.array) -> np.array:
        super().basis_function__(x_)
        fb: np.array = np.array([])
        for _mu_i in self.mm__:
            _r_i = gaussian_multivariate_2d(x_, _mu_i, self.cov_inv)
            fb = np.append(fb, _r_i)
        fb = fb[:, np.newaxis]
        # Verify shapes
        ShapeChecker.check_shape(fb, (self.f, 1))
        return fb


class Plotter:
    def __init__(self, model: Model2D):
        self.model: Model2D = model
        # Plot
        self.fig: Figure = plt.figure(figsize=(15, 5))

        self.ax_error = self.fig.add_subplot(244)
        self.ax_line_error: Line2D = self.ax_error.plot([], [], label='Error', c='r')[0]
        self.queue_error = self.model.queue_error

        self.ax_points = self.fig.add_subplot(248, projection='3d')

        self.ax_model = self.fig.add_subplot(121, projection='3d')
        self.ax_model_data = None

    def plot(self):
        self.fig.suptitle('Gaussian Model')

        self.ax_points.scatter(self.model.data_x[0], self.model.data_x[1], self.model.data_y, label='Data')
        self.ax_points.set_title('Data')
        self.ax_points.set_xlabel('x_0')
        self.ax_points.set_ylabel('x_1')
        self.ax_points.set_zlabel('y')
        self.ax_points.legend()

        self.ax_error.set_title('Error')
        self.ax_error.set_xlabel('Epoch')
        self.ax_error.set_ylabel('Error')
        self.ax_error.legend()

        self.ax_model.scatter(self.model.data_x[0], self.model.data_x[1], self.model.data_y, label='Data')

        _x_set_plot: np.array = PlotterData.get_x_plot_2d()
        _y_set_plot: np.array = self.model.g(_x_set_plot)

        self.ax_model.plot_trisurf(_x_set_plot[0], _x_set_plot[1], _y_set_plot, cmap='viridis', edgecolor='none')

        self.ax_model.set_title('Model')
        self.ax_model.set_xlabel('x_0')
        self.ax_model.set_ylabel('x_1')
        self.ax_model.set_zlabel('y')
        # self.ax_model.set_xlim(self.model.data_limits[0], self.model.data_limits[1])
        # self.ax_model.set_ylim(self.model.data_limits[0], self.model.data_limits[1])
        # self.ax_model.set_zlim(-2, 2)
        self.ax_model.legend()

        # create animation using the animate() function
        _ani = FuncAnimation(
            self.fig,
            self.plot_callback,
            blit=True,
            interval=1000  # for 60 fps use interval=17
        )

        plt.tight_layout()
        plt.show()

    def plot_callback(self, frame):
        return self.plot_callback_error(),

    def plot_callback_error(self):

        _new_data: np.array = np.array([])
        while not self.queue_error.empty():
            error = self.queue_error.get()
            print(error)
            _new_data = np.append(_new_data, error)

        if _new_data.shape[0] > 0:
            self.model.error_history = np.append(self.model.error_history, _new_data)
            _xdata: np.array = np.arange(self.model.error_history.shape[0]) + 1
            self.ax_line_error.set_xdata(_xdata)
            self.ax_line_error.set_ydata(self.model.error_history)

            self.ax_error.relim()
            self.ax_error.autoscale_view()
            self.fig.canvas.draw()

        return self.ax_line_error
    # def plot_callback_error(self):
    #     _new_ydata: np.array = np.array([])
    #     while not self.queue_error.empty():
    #         error = self.queue_error.get()
    #         print(error)
    #         _new_ydata = np.append(_new_ydata, error)
    #
    #     if _new_ydata.shape[0] > 0:
    #         _ydata: np.array = self.ax_line_error.get_ydata()
    #         _ydata = np.append(_ydata, _new_ydata)
    #         _xdata: np.array = np.arange(_ydata.shape[0]) + 1
    #         self.ax_line_error.set_xdata(_xdata)
    #         self.ax_line_error.set_ydata(_ydata)
    #
    #         self.ax_error.relim()
    #         self.ax_error.autoscale_view()
    #         self.fig.canvas.draw()
    #
    #     return self.ax_line_error


if __name__ == '__main__':
    model_g: Model2D = Model2DGaussian(factors_x_dim=5, epochs=50, a=0.1, _s2=0.1)
    model_g.summary()

    # model_g.train_callback()
    model_g.train()

    plotter: Plotter = Plotter(model_g)
    plotter.plot()
    plotter.model.process.join()
