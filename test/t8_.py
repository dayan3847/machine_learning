import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

fig: Figure = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
line_2d, = ax.plot([], [], label='Error', c='r')
ax.set_title('Error')
ax.set_xlabel('Epoch')
ax.set_ylabel('Error')
# ax.set_xlim(left=0)
# ax.set_ylim(bottom=0)
ax.legend()


# Función de actualización para la animación
def update(frame):
    print(frame)
    y_ran = np.random.uniform()
    new_x_data = np.append(line_2d.get_xdata(), frame)
    new_y_data = np.append(line_2d.get_ydata(), y_ran)

    # Actualizar los datos
    line_2d.set_xdata(new_x_data)
    line_2d.set_ydata(new_y_data)

    # Ajustar los límites de los ejes x e y
    ax.relim()
    ax.autoscale_view()

    # actualizar ejes
    fig.canvas.draw()

    return line_2d,


ani = FuncAnimation(fig, update, blit=True, interval=1000)

plt.tight_layout()
plt.show()
