import matplotlib.pyplot as plt
import numpy as np
import random

plt.ion() # habilita la modalidad interactiva

def update_plot(i):
    # Generar nuevos datos
    y = np.random.randn(100)
    # Limpiar la figura y redibujar
    plt.clf()
    plt.plot(y)
    plt.draw()

for i in range(100):
    update_plot(i)
    plt.pause(0.5)

plt.ioff() # desactiva la modalidad interactiva
plt.show()
