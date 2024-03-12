import numpy as np

a: float = 0.1

w1 = np.random.rand(4, 2)


def propagate_w1(input_: np.array):
    return np.append(input_ @ w1, 1)


w2 = np.random.rand(3, 10)


def propagate_w2(input_: np.array):
    return input_ @ w2


def propagate(input_: np.array):
    # return propagate_w2(propagate_w1(input_))
    return np.append(input_ @ w1, 1) @ w2


y = np.random.rand(10)

x = np.append(np.random.rand(3), 1)


def retro_propagate(x: np.array, y: np.array):
    global w1, w2
    o1 = np.append(x @ w1, 1)
    print('o1', o1.shape)
    o2 = o1 @ w2
    d2 = np.multiply(o2, np.multiply(1 - o2, y - o2))
    # output neuron:
    w2 += a * np.outer(o1, d2)
    # hidden neuron:
    wdwdwd = []
    for i in range(w2.shape[0]):
        wd_i = 0
        for j in range(w2.shape[1]):
            wd_i += w2[i, j] * d2[j]
        wdwdwd.append(wd_i)
    print('wdwdwd', np.array(wdwdwd).shape)
    print('1 - o1', (1 - o1).shape)
    print('np.array(wdwdwd)', np.array(wdwdwd))

    qwe = np.multiply(1 - o1, np.array(wdwdwd))
    print('qwe', qwe.shape)
    d1 = np.multiply(o1, qwe)
    d1 = d1[0:-1]
    print('d1', d1.shape)
    w1 += a * np.outer(x, d1)


retro_propagate(x, y)
