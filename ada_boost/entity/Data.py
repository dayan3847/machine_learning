import numpy as np
from typing import List


class Data:
    def __init__(self, x: List[np.ndarray], target: np.ndarray, weights: np.ndarray, count: int):
        self.x: List[np.ndarray] = x
        self.target: np.ndarray = target
        self.weights: np.ndarray = weights
        self.count: int = count

    def normalize(self):
        # normalizar los datos de entrada de 0 a 100
        for f in range(len(self.x)):
            self.x[f] = (self.x[f] - self.x[f].min()) / (self.x[f].max() - self.x[f].min()) * 100

    def normalize_weights(self):
        sum_weights = self.weights.sum()
        self.weights = self.weights / sum_weights

    @staticmethod
    def load_data() -> 'Data':
        x: List[np.ndarray] = [
            np.array([]),  # x0
            np.array([]),  # x1
        ]
        # target
        target: np.ndarray = np.array([])

        # cargar de un archivo
        with open('corpus/dataCircle.txt', 'r') as f:
            for line in f:
                nums = line.split()
                # coordinate x0 (dimension 0)
                x[0] = np.append(x[0], float(nums[0]))
                # coordinate x1 (dimension 1)
                x[1] = np.append(x[1], float(nums[1]))
                # target
                target = np.append(target, float(nums[2]))

        count = len(target)
        # pesos
        weights = np.ones(count) / count

        return Data(x, target, weights, count)
