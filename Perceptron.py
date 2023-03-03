from typing import List
import numpy as np


class Perceptron:
    my_weights: List[float]

    def __init__(self, weights_count: int = 0):
        self.weights = self.generate_weights(weights_count) if 0 < weights_count else []

    def set_weights(self, weights: List[float]):
        self.weights = weights

    def net_input_numpy(self, features: List[float]) -> float:
        # b is weights[0], features[0] is 1
        features_copy: List[float] = features.copy()
        features_copy.insert(0, 1)
        return float(np.dot(features_copy, self.weights))

    def net_input_teacher(self, features: List[float]) -> float:
        result: float = self.weights[0]
        count: int = len(self.weights)
        for i in range(1, count):
            result += features[i - 1] * self.weights[i]
        return result

    @staticmethod
    def activation_function(x: float) -> float:  # sigmoid
        return 1 / (1 + np.exp(-x))

    def propagate(self, features: List[float]) -> float:
        sigma: float = self.net_input_teacher(features)
        return self.activation_function(sigma)

    @staticmethod
    def generate_weights(count: int) -> List[float]:
        return [np.random.uniform(-.5, .5) for _ in range(count)]
