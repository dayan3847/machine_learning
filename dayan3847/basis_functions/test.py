import numpy as np
x_plot: np.array = np.array([1, 2, 3])
y_plot: np.array = np.array([4, 5, 6])

asd = np.meshgrid(x_plot, y_plot)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

print(X_plot.flatten())
print(Y_plot.flatten())

print(X_plot.ravel())
print(Y_plot.ravel())

