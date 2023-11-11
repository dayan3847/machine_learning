import numpy as np

from dayan3847.models.Model import Model


class ModelTrainer:
    def __init__(self,
                 model: Model,
                 epochs_count: int,
                 data_x: np.array,
                 data_y: np.array,
                 ):
        self.model: Model = model
        self.epoch: int = 0
        self.epochs_count: int = epochs_count
        self.data_x: np.array = data_x
        self.data_y: np.array = data_y
        self.error_history: np.array = np.array([])  # Error history

    def e(self) -> float:
        g: np.array = self.model.g(self.data_x)
        y: np.array = self.data_y
        dif: np.array = g - y
        return np.sum(dif ** 2) / 2

    def train(self):
        while self.epoch < self.epochs_count:
            self.train_step()

    def train_step(self):
        self.update_w_for_all_dataset()
        self.save_current_error()
        self.epoch += 1
        print('epoch: {}/{} error: {}'.format(
            self.epoch,
            self.epochs_count,
            self.e()
        ))

    def update_w_for_all_dataset(self):
        self.model.update_w(self.data_x, self.data_y)

    def save_current_error(self):
        self.error_history = np.append(self.error_history, self.e())
