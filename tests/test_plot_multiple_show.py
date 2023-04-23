import matplotlib.pyplot as plt

# Creamos la figura y los ejes
fig, ax = plt.subplots()

# Hacemos el primer plot
ax.plot([0], [0], 'ro')  # Punto rojo en (0,0)

# Mostramos el primer plot
fig.show()

# Hacemos el segundo plot
ax.plot([1], [1], 'bo')  # Punto azul en (1,1)

# Mostramos el segundo plot, manteniendo la figura abierta
fig.show()

ax.clear()

# Plot de una linea vertical
ax.axvline(x=0.5, color='green')

# Mostramos el plot
fig.show()
