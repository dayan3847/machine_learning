import numpy as np


class Normalizer:

    @staticmethod
    def normalize_vector_points(vector: np.ndarray, r_to: float = 1, r_from: float = 0) -> np.ndarray:
        min_val = np.min(vector)
        max_val = np.max(vector)
        normalized_vector = np.interp(vector, (min_val, max_val), (r_from, r_to))
        return normalized_vector
