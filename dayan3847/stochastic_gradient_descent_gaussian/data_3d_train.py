import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import threading
from scipy.stats import multivariate_normal
from abc import ABC, abstractmethod

from matplotlib import animation


class ShapeChecker:

    @staticmethod
    def check_shape(_array: np.array, _expected_shape: tuple):
        if _array.shape != _expected_shape:
            raise Exception('Expected shape: {}, actual shape: {}'.format(_expected_shape, _array.shape))

    @staticmethod
    def check_shape_data(_data: np.array):
        if len(_data.shape) != 2 or _data.shape[0] != 3:
            raise Exception('Expected shape: (3, N), actual shape: {}'.format(_data.shape))


class Model2D(ABC):
    def __init__(self, cxd: int = 3, epochs: int = 1, a: float = 0.1):
        self.a: float = a  # Learning rate
        self.cxd: int = cxd  # Number of basis functions for dimensions
        self.epochs: int = epochs  # Number of epochs

        # example shape(3,1000,)
        # 1000 cantidad de ejemplos
        # de los 3, el ultimo de Y y los anteriores forman el vector X
        self.data: np.array = np.loadtxt('data_3d.csv', delimiter=',').T  # Load Data
        self.n: int = self.data.shape[1]  # Examples count
        # dimensions of x
        self.dim_x: int = self.data.shape[0] - 1  # Dimensions count
        self.data_limits: tuple = (0, 1)

        self.f: int = self.cxd ** 2  # Number of factors
        # Weights Ex: shape(25,)
        self.w__: np.array = np.random.rand(self.f) - .5  # Weights
        # aumentar el peso del medio
        self.w__[self.f // 2] = 5
        self.a__: np.array = np.full(self.f, self.a)  # Learning rate for each weight

        self.error_history: np.array = np.array([])  # Error history
        self.thread: threading.Thread = threading.Thread(target=self.train_callback)

        # Check shapes
        ShapeChecker.check_shape_data(self.data)
        ShapeChecker.check_shape(self.w__, (self.f,))
        ShapeChecker.check_shape(self.a__, (self.f,))

    @abstractmethod
    def equation_basis_function(self) -> sp.Expr:
        pass

    # @abstractmethod
    # def equation(self) -> sp.Expr:
    #     pass

    # Calculate the basis function a partir de un valor X
    # retorna un vector de shape(25,)
    # cada valor del vector es el resultado de aplicar la funcion base que se multiplica por cada peso
    # entrada esperada: Ex: shape(25,2) donde 25 es la cantidad de funciones base y 2 es la cantidad de dimensiones de X
    def basis_function__(self, _x_i: np.array) -> np.array:
        ShapeChecker.check_shape(_x_i, (self.dim_x,))

    # Calculate the model value for a simple value

    # entrada esperada: Ex: shape(2,) donde 2 es la cantidad de dimensiones de X
    # Expected shape: (x_dim,)
    def hi(self, xi: np.array) -> float:
        # Check shapes
        ShapeChecker.check_shape(xi, (self.dim_x,))
        bf: np.array = self.basis_function__(xi)
        # Check shapes
        ShapeChecker.check_shape(bf, (self.f,))
        return np.dot(self.w__, bf)

    def activate(self, h: float) -> float:
        # return 1 / (1 + np.exp(-h))
        return h

    # Expected shape: (x_dim,)
    def gi(self, xi: np.array) -> float:
        # Check shapes
        ShapeChecker.check_shape(xi, (self.dim_x,))
        return self.activate(self.hi(xi))

    # Expected shape: (N, x_dim,)
    def g(self, x: np.array) -> np.array:
        # Check shapes
        # ShapeChecker.check_shape(x, (self.n, self.dim_x))
        return np.array([self.gi(_xi) for _xi in x])

    # def classify_i(self, x: float, y: float) -> int:
    #     return int(round(self.gi(x, y)))
    #
    # def classify(self, x: np.array, y: np.array) -> np.array:
    #     return np.array([self.classify_i(xi, yi) for xi, yi in zip(x, y)])

    def e(self) -> float:
        return np.sum((self.g(self.data[0:-1].T) - self.data[-1]) ** 2) / 2

    # def accuracy(self) -> float:
    #     return np.sum(self.classify(self.data[0], self.data[1]) == self.data[2]) / self.data.shape[1]

    def summary(self):
        print('Model: {}'.format(self.__class__.__name__))
        print('Error: {}'.format(round(self.e(), 2)))
        # print('Accuracy: {}'.format(round(self.accuracy(), 2)))

    def train_callback(self):
        for _ in range(self.epochs):
            self.save()
            self.train_step()

    def train(self):
        self.thread.start()

    def train_step(self):
        xx_: np.array = self.data[0:-1].T
        y_: np.array = self.data[-1]

        for x_i, y_i in zip(xx_, y_):
            b__: np.array = self.basis_function__(x_i)
            g_i: float = self.gi(x_i)
            g_i__: np.array = np.full(self.f, g_i)
            y_i__: np.array = np.full(self.f, y_i)
            dw__: np.array = self.a__ * (g_i__ - y_i__) * b__
            # Verify shapes
            ShapeChecker.check_shape(dw__, (self.f,))
            print(dw__)
            self.w__ -= dw__

    def save(self):
        self.error_history = np.append(self.error_history, self.e())


class Model2DGaussian(Model2D):

    def __init__(self, cxd: int = 3, epochs: int = 10, a: float = 0.1):
        super().__init__(cxd, epochs, a)
        _possible_m: np.array = np.linspace(self.data_limits[0], self.data_limits[1], self.cxd)
        _d: int = self.dim_x
        self.mm__: np.array = np.array(np.meshgrid(*[_possible_m] * _d)).T.reshape(-1, _d)
        _s: float = 10.
        # covariance matrix identity
        self.cov: np.array = np.identity(self.dim_x) * _s
        self.cov_inv: np.array = np.linalg.inv(self.cov)
        # Verify shapes
        ShapeChecker.check_shape(self.mm__, (self.f, self.dim_x))
        ShapeChecker.check_shape(self.cov, (self.dim_x, self.dim_x))

    def equation_basis_function(self) -> sp.Expr:
        _X: sp.Symbol = sp.MatrixSymbol('X', self.dim_x, 1)
        # mu mayuscula
        _M: sp.MatrixSymbol = sp.MatrixSymbol('mu', self.dim_x, 1)
        # sigma mayuscula
        _C: sp.MatrixSymbol = sp.MatrixSymbol('sigma', self.dim_x, self.dim_x)
        return sp.exp(-.5 * (_X - _M).T * _C.I * (_X - _M))

    # def equation(self) -> sp.Expr:
    #     r: sp.Symbol = 0
    #     for w, m1, m2, s in zip(self.w, self.m1, self.m2, self.s):
    #         w_: float = round(w, 2)
    #         r += w_ * self.equation_basis_function().subs({'m1': round(m1, 2), 'm2': round(m2, 2), 's': s})
    #     return r

    def basis_function__(self, _x_i: np.array) -> np.array:
        super().basis_function__(_x_i)

        _R: np.array = np.zeros(self.f)
        # Convert to column vector
        # _Xic: np.array = _x_i.reshape((self.dim_x, 1))
        for i in range(self.f):
            _mu: np.array = self.mm__[i]
            _rv = multivariate_normal(_mu, self.cov)
            _R[i] = _rv.pdf(_x_i)
            # _M: np.array = self.mm__[i].reshape((self.dim_x, 1))
            # _X_M = _Xic - _M
            # _X_MT_Ci = np.dot(_X_M.T, _Ci)
            # _X_MT_Ci_X_M = np.dot(_X_MT_Ci, _X_M)
            # x_m_t_ci_x_m = _X_MT_Ci_X_M[0][0]
            # r = np.exp(-.5 * x_m_t_ci_x_m)
            # _R[i] = r

        return _R


class Plotter:
    def __init__(self, model: Model2D):
        self.model: Model2D = model
        # Plot
        self.fig, _ = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
        self.fig.suptitle('Gaussian Model')
        self.ax_error = None
        self.ax_error_data = None
        self.ax_points = None
        self.ax_model = None
        self.ax_model_data = None

    def plot(self):
        plt.cla()
        plt.clf()

        self.ax_points = plt.subplot(248, projection='3d')
        self.ax_points.scatter(self.model.data[0], self.model.data[1], self.model.data[2], label='Data')
        self.ax_points.set_title('Data')
        self.ax_points.set_xlabel('x_0')
        self.ax_points.set_ylabel('x_1')
        self.ax_points.set_zlabel('y')
        self.ax_points.legend()

        self.ax_error = plt.subplot(244)
        self.ax_error_data = self.ax_error.plot(self.model.error_history, label='Error', c='r')
        self.ax_error.set_title('Error')
        self.ax_error.set_xlabel('Epoch')
        self.ax_error.set_ylabel('Error')
        self.ax_error.set_xlim(left=0)
        self.ax_error.set_ylim(bottom=0)
        self.ax_error.legend()

        self.ax_model = plt.subplot(121, projection='3d')
        self.ax_model.scatter(self.model.data[0], self.model.data[1], self.model.data[2], label='Data')

        _num = 20
        _x0 = np.linspace(self.model.data_limits[0], self.model.data_limits[1], _num)
        _x1 = np.linspace(self.model.data_limits[0], self.model.data_limits[1], _num)
        _x0, _x1 = np.meshgrid(_x0, _x1)
        _x0 = _x0.flatten()
        _x1 = _x1.flatten()
        _y = self.model.g(np.array([_x0, _x1]).T)
        _x0 = _x0.reshape((_num, _num))
        _x1 = _x1.reshape((_num, _num))
        _y = _y.reshape((_num, _num))
        self.ax_model.plot_surface(_x0, _x1, _y, alpha=0.5, label='Model')

        self.ax_model.set_title('Model')
        self.ax_model.set_xlabel('x_0')
        self.ax_model.set_ylabel('x_1')
        self.ax_model.set_zlabel('y')
        self.ax_model.set_xlim(self.model.data_limits[0], self.model.data_limits[1])
        self.ax_model.set_ylim(self.model.data_limits[0], self.model.data_limits[1])
        # self.ax_model.set_zlim(-2, 2)
        # self.ax_model.legend()

        # create animation using the animate() function
        _ani = animation.FuncAnimation(self.fig, self.plot_callback, interval=17)  # 60 fps

        plt.tight_layout()
        plt.show()

    def plot_callback(self, frame):
        # self.ax_error_data[0].set_ydata(self.model.error_history)
        self.ax_error_data[0].set_ydata([frame])
        self.ax_error.relim()
        self.ax_error.autoscale_view()
        self.fig.tight_layout()


# def plot_model_2d_in_3d(model_: Model2D) -> None:
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlim(model_.data_limits[0], model_.data_limits[1])
#     ax.set_ylim(model_.data_limits[0], model_.data_limits[1])
#     ax.set_zlim(-2, 2)
#
#     ax.set_title('Model')
#
#     _data: np.ndarray = model_.data
#
#     ax.scatter(_data[0, _data[2, :] == 0], _data[1, _data[2, :] == 0], _data[2, _data[2, :] == 0], c='red', marker='o')
#     ax.scatter(_data[0, _data[2, :] == 1], _data[1, _data[2, :] == 1], _data[2, _data[2, :] == 1], c='green',
#                marker='o')
#
#     _x0 = np.linspace(model_.data_limits[0], model_.data_limits[1], 100)
#     _x1 = np.linspace(model_.data_limits[0], model_.data_limits[1], 100)
#     _x0, _x1 = np.meshgrid(_x0, _x1)
#     _x0 = _x0.flatten()
#     _x1 = _x1.flatten()
#     _y = model_.g(np.array([_x0, _x1]).T)
#     _x0 = _x0.reshape((100, 100))
#     _x1 = _x1.reshape((100, 100))
#     _y = _y.reshape((100, 100))
#     ax.plot_surface(_x0, _x1, _y, alpha=0.5)
#
#     plt.show()


if __name__ == '__main__':
    model_g: Model2D = Model2DGaussian()
    model_g.summary()
    print(model_g.equation_basis_function())

    # model_g.train()

    plotter: Plotter = Plotter(model_g)
    plotter.plot()
