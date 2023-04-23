import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)

# Dibuja la línea vertical en x=3
plt.axvline(x=3, color='red')

# Dibuja la línea horizontal en y=3
plt.axhline(y=3, color='green')


plt.show()
