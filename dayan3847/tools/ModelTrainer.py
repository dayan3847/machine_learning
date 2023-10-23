import numpy as np
import threading

from dayan3847.tools import Model, ShapeChecker


class ModelTrainer:
    def __init__(self, model_: Model, data_: np.array, epochs_: int):
        self.model: Model = model_
        self.epochs: int = epochs_  # Number of epochs
        self.running: bool = False
        self.current_epoch: int = 0

        ShapeChecker.check_shape_point_set(data_, 3)
        self.data_x: np.array = data_[0:2]
        self.data_y: np.array = data_[2]
        self.n: int = data_.shape[1]  # Examples count

        self.error_history: np.array = np.array([])  # Error history
        # self.error_history: np.array = np.array([self.e()])  # Error history
        self.thread: threading.Thread = threading.Thread(target=self.train_callback)

    def e(self) -> float:
        _g_set: np.array = self.model.g(self.data_x)
        _y_set: np.array = self.data_y
        _diff_set: np.array = _g_set - _y_set
        _e: float = np.sum(_diff_set ** 2) / 2
        return _e

    def train_callback(self):
        while self.current_epoch < self.epochs and self.running:
            self.train_step()
            print('epoch: {} error: {}'.format(self.current_epoch, self.e()))

    def train_step(self):
        self.update_w_for_all_dataset()
        self.save_current_error()
        self.current_epoch += 1

    def train(self):
        self.thread.start()

    def update_w_for_all_dataset(self):
        for _x_1d, _y in zip(self.data_x.T, self.data_y):
            self.model.update_w(_x_1d, _y)

    def save_current_error(self):
        self.error_history = np.append(self.error_history, self.e())
