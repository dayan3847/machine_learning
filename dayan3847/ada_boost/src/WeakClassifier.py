import numpy as np


class WeakClassifier:
    def __init__(self, feature: int, threshold: int, polarity: bool = True):
        self.feature: int = feature
        self.threshold: int = threshold
        self.polarity: bool = polarity
        self.error: float | None = None
        self.alpha: float | None = None

    # ** Clasificar un conjunto de ejemplos
    # * @param examples El primer índice es la dimension o feature y el segundo el ejemplo (np.ndarray)
    def classify(self, points_t: np.ndarray) -> np.ndarray:
        feature_value: np.ndarray = points_t[self.feature]
        right_value: int = 1 if self.polarity else -1
        left_value: int = -1 * right_value
        return np.where(feature_value < self.threshold, left_value, right_value)

    def get_alpha(self, data_t: np.ndarray) -> float:
        if self.alpha is None:
            e = self.get_error(data_t)
            self.alpha = .5 * np.log((1 - e) / e)
        return self.alpha

    def get_error(self, data: np.ndarray, fix_polarity: bool = True) -> float:
        self.error = self._get_error(data)
        if self.error > .5 and fix_polarity:
            self.polarity = not self.polarity
            self.error = 1 - self.error
        return self.error

    # ** Calcular el error de clasificación
    def _get_error(self, data: np.ndarray) -> float:
        if self.error is None:
            self.error = 0
            classification = self.classify(data)
            for i in range(data.shape[1]):
                if classification[i] != data[2][i]:
                    self.error += data[3][i]
        return self.error
