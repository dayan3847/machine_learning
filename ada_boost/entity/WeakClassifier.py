import numpy as np
from typing import List

from ada_boost.entity.Data import Data


class WeakClassifier:
    def __init__(self, feature: int, threshold: int, polarity: bool = True):
        self.feature: int = feature
        self.threshold: int = threshold
        self.polarity: bool = polarity
        self.error: float | None = None
        self.alpha: float | None = None

    # ** Clasificar un conjunto de ejemplos
    # * @param examples El primer índice es la dimension o feature y el segundo el ejemplo (np.ndarray)
    def classify(self, points: List[np.ndarray]) -> np.ndarray:
        feature_value: np.ndarray = points[self.feature]
        right_value: int = 1 if self.polarity else -1
        left_value: int = -1 * right_value
        return np.where(feature_value < self.threshold, left_value, right_value)

    # ** Calcular el error de clasificación
    def get_error(self, data: Data) -> float:
        if self.error is None:
            self.error = 0
            classification = self.classify(data.x)
            for i in range(data.count):
                if classification[i] != data.target[i]:
                    self.error += data.weights[i]
        return self.error

    def get_alpha(self, data: Data) -> float:
        if self.alpha is None:
            e = self.fix_polarity(data)
            self.alpha = .5 * np.log((1 - e) / e)
        return self.alpha

    def fix_polarity(self, data: Data) -> float:
        self.error = self.get_error(data)
        if self.error < .5:
            self.polarity = not self.polarity
            self.error = 1 - self.error
        return self.error
