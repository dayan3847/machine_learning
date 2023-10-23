import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Función para actualizar el gráfico
def update_graph(i):
    # Generar datos o realizar cálculos aquí
    x = np.linspace(0, 10, 100)
    y = np.sin(x + i * 0.1)  # Ejemplo: una función seno que cambia con el tiempo

    # Limpiar la figura anterior y dibujar el nuevo gráfico
    plt.clf()
    plt.plot(x, y)
    plt.title("Gráfico Dinámico")
    plt.xlabel("X")
    plt.ylabel("Y")

# Crear una figura de Matplotlib
fig = plt.figure()

# Crear una animación que actualiza el gráfico cada segundo
ani = FuncAnimation(fig, update_graph, interval=1000)  # 1000 ms (1 segundo)

# Mostrar la ventana gráfica
plt.show()

# Esperar hasta que se cierre la ventana (puede interrumpir la ejecución)
cv2.waitKey(0)
cv2.destroyAllWindows()
