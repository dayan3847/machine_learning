import numpy as np

from dayan3847.models.Model import Model


def get_model_error(model: Model,
                    data_x: np.array,
                    data_y: np.array,
                    ):
    g: np.array = model.g(data_x)
    y: np.array = data_y
    dif: np.array = g - y
    return np.sum(dif ** 2) / 2


def train_model(model: Model,
                data_x: np.array,
                data_y: np.array,
                a: float,
                epochs_count: int = 1,
                error_threshold: float | None = None,
                v: bool = True,  # verbose
                ) -> list[float]:  # Error history
    epoch: int = 0

    e: float = get_model_error(model, data_x, data_y)
    if v:
        print('epoch: {}/{} error: {}'.format(0, epochs_count, e))
    error_history: list[float] = [e]  # Error history
    if error_threshold is not None and e < error_threshold:
        return error_history

    while epoch < epochs_count:
        epoch += 1
        model.train(data_x, data_y, a)
        e: float = get_model_error(model, data_x, data_y)
        if v:
            print('epoch: {}/{} error: {}'.format(epoch, epochs_count, e))
        error_history.append(e)
        if error_threshold is not None and e < error_threshold:
            return error_history

    return error_history
