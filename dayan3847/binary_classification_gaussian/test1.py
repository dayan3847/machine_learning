# def generate_dataset():
import numpy as np

_data_0: np.ndarray = np.ndarray(shape=(2, 0), dtype=np.float32)
_data_1: np.ndarray = np.ndarray(shape=(2, 0), dtype=np.float32)
# generate class 0
while _data_0.shape[1] < N / 2:
    _x = np.random.uniform(0, S)
    _y = np.random.uniform(0, S)
    if C.is_in_point_any(vc, _x, _y):
        if _data_1.shape[1] < N / 2:
            _data_1 = np.append(_data_1, np.array([[_x], [_y]]), axis=1)
    else:
        _data_0 = np.append(_data_0, np.array([[_x], [_y]]), axis=1)
# generate class 1
while _data_1.shape[1] < N / 2:
    _c = np.random.choice(vc)
    _r = np.random.uniform(0, _c.r)
    _a = np.random.uniform(0, 2 * np.pi)
    _x = _c.x + _r * np.cos(_a)
    _y = _c.y + _r * np.sin(_a)
    _data_1 = np.append(_data_1, np.array([[_x], [_y]]), axis=1)

_data: np.ndarray = np.append(_data_0, np.zeros(shape=(1, _data_0.shape[1]), dtype=np.float32), axis=0)
_data = np.append(_data, np.append(_data_1, np.ones(shape=(1, _data_1.shape[1]), dtype=np.float32), axis=0), axis=1)


def plot_circ(data_: np.ndarray) -> None:
    fig, ax = plt.subplots()
    ax.set_xlim(0, S)
    ax.set_ylim(0, S)
    ax.plot(data_[0, data_[2, :] == 0], data_[1, data_[2, :] == 0], 'o', color='blue')
    ax.plot(data_[0, data_[2, :] == 1], data_[1, data_[2, :] == 1], 'o', color='red')
    for c in vc:
        ax.add_patch(plt.Circle((c.x, c.y), c.r, color='green', fill=False))
    plt.show()