from typing import List
import numpy as np


class Perceptron:
    weight_features: List[float]
    weight_bias: float

    def __init__(self, weights_count: int = 0):
        if 0 < weights_count:
            self.weight_bias = self.generate_weight()
            self.weight_features = [self.generate_weight() for _ in range(weights_count)]

    def set_weights(self, weights: List[float]):
        weights_copy: List[float] = weights.copy()
        self.weight_bias = weights_copy.pop(0)
        self.weight_features = weights_copy

    def net_input_numpy(self, features: List[float]) -> float:
        features_length: int = len(features)
        if features_length != len(self.weight_features):
            raise Exception(f'Features length {features_length} != weights length {len(self.weight_features)}')
        return float(np.dot(features, self.weight_features)) + self.weight_bias

    def net_input_teacher(self, features: List[float]) -> float:
        features_length: int = len(features)
        if features_length != len(self.weight_features):
            raise Exception(f'Features length {features_length} != weights length {len(self.weight_features)}')
        result: float = self.weight_bias
        for i in range(features_length):
            result += features[i] * self.weight_features[i]
        return result

    def activation_function(self, x: float) -> float:  # sigmoid
        return 1 / (1 + np.exp(-x))

    def propagate(self, features: List[float]) -> float:
        sigma: float = self.net_input_teacher(features)
        return self.activation_function(sigma)

    def generate_weight(self) -> float:
        return np.random.uniform(-.5, .5)

    def generate_weights(self, count: int) -> List[float]:
        return [self.generate_weight() for _ in range(count)]
