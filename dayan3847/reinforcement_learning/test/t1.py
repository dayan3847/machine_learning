import sys
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem
from PyQt5.QtCore import QTimer

class DynamicGraphWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Crear una vista gráfica y una escena
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        # Configurar la ventana principal
        self.setCentralWidget(self.view)
        self.setGeometry(100, 100, 400, 400)
        self.setWindowTitle("Gráfico Dinámico")

        # Inicializar un temporizador para actualizar el gráfico cada segundo
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_graph)
        self.timer.start(1000)  # Actualizar cada 1000 ms (1 segundo)

        # Agregar un círculo inicial al gráfico
        self.circle = QGraphicsEllipseItem(0, 0, 50, 50)
        self.scene.addItem(self.circle)

    def update_graph(self):
        # Actualizar la posición del círculo con valores aleatorios
        new_x = random.randint(0, 350)
        new_y = random.randint(0, 350)
        self.circle.setRect(new_x, new_y, 50, 50)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DynamicGraphWindow()
    window.show()
    sys.exit(app.exec_())
