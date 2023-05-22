import numpy as np
from sklearn.model_selection import train_test_split


class TrainTestValSpliter:

    @staticmethod
    def train_test_val_spliter(data: np.ndarray, test_size: float = .3, val_size: float = 0):
        data_train, data_test = train_test_split(data.T, test_size=test_size + val_size, random_state=42)
        if val_size == 0:
            return data_train.T, data_test.T
        data_test, data_val = train_test_split(data_test, test_size=val_size, random_state=42)
        return data_train.T, data_test.T, data_val.T
