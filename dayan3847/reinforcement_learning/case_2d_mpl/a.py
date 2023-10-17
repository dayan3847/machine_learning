# import numpy as np
# import matplotlib.pyplot as plt
#
# np.random.seed(19680801)
# Z = np.random.rand(6, 10)
# x = np.arange(-0.5, 10, 1)  # len = 11
# y = np.arange(4.5, 11, 1)  # len = 7
#
# fig, ax = plt.subplots()
# ax.pcolormesh(x, y, Z)


import matplotlib.pyplot as plt
import numpy as np

green = (0, 255, 0)
gray = (128, 128, 128)
black = (0, 0, 0)
red = (255, 255, 0)
blue1 = (0, 0, 128)

env = np.array([
    [gray, gray, gray, gray, gray, gray, gray, gray, gray, gray, gray, gray],
    [gray, gray, gray, gray, gray, gray, gray, gray, gray, gray, gray, gray],
    [gray, gray, gray, gray, gray, gray, gray, gray, gray, gray, gray, gray],
    [blue1, black, black, black, black, black, black, black, black, black, black, green],
])
init_pos = (3, 0)
env[init_pos] = red

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[6, 3])
im1 = ax1.imshow(env)
fig.colorbar(im1)
plt.show()
