import matplotlib.pyplot as plt

# Datos
categorias = ['Categoría 1', 'Categoría 2', 'Categoría 3', 'Categoría 4']
valores = [.10, .25, .15, .4]

# y entre 0 y 1
plt.ylim(0, 1)

# Crear el gráfico de barras
plt.bar(categorias, valores)

# Etiquetas y título
plt.xlabel('Categorías')
plt.ylabel('Valores')
plt.title('Gráfico de Barras Personalizado')

# Mostrar el gráfico
plt.show()
