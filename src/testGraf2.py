import matplotlib.pyplot as plt
import numpy as np
import random

plt.ion() # habilita la modalidad interactiva

fig, ax = plt.subplots()
x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

def update(frame):
    line.set_ydata(np.sin(x + frame/10.0)+random.random())  # actualiza los datos en cada frame
    return line,

for i in range(100):
    update(i)
    plt.pause(0.05)

plt.ioff() # desactiva la modalidad interactiva
plt.show()
