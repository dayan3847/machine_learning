class Model2D(ABC):
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data  # Data
        self.cxd: int = 5  # Number of basis functions for dimensions
        self.c: int = self.cxd ** 2  # Number of basis functions
        self.w = np.random.rand(self.c) - .5  # Weights
        self.epochs: int = 1000  # Number of epochs
        self.a: float = 0.1  # Learning rate

        self.error_history: np.array = np.array([])  # Error history

    @abstractmethod
    def equation_basis_function(self) -> sp.Expr:
        pass

    @abstractmethod
    def basis_function(self, x: float, y: float) -> np.array:
        pass

    @abstractmethod
    def equation(self) -> sp.Expr:
        pass

    # Calculate the model value for a simple value
    def hi(self, x: float, y: float) -> float:
        bf: np.array = self.basis_function(x, y)
        return np.dot(self.w, bf)

    # Calculate the model value for a vector
    def h(self, x: np.array, y: np.array) -> np.array:
        return np.array([self.hi(xi, yi) for xi, yi in zip(x, y)])

    def activate(self, h: float) -> float:
        # return 1 / (1 + np.exp(-h))
        return h

    def gi(self, x: float, y: float) -> float:
        return self.activate(self.hi(x, y))

    def g(self, x: np.array, y: np.array) -> np.array:
        return np.array([self.gi(xi, yi) for xi, yi in zip(x, y)])

    def classify_i(self, x: float, y: float) -> int:
        return int(round(self.gi(x, y)))

    def classify(self, x: np.array, y: np.array) -> np.array:
        return np.array([self.classify_i(xi, yi) for xi, yi in zip(x, y)])

    def e(self) -> float:
        return np.sum((self.g(self.data[0], self.data[1]) - self.data[2]) ** 2) / 2

    def accuracy(self) -> float:
        return np.sum(self.classify(self.data[0], self.data[1]) == self.data[2]) / self.data.shape[1]

    def summary(self):
        print('Model: {}'.format(self.__class__.__name__))
        print('Error: {}'.format(round(self.e(), 2)))
        print('Accuracy: {}'.format(round(self.accuracy(), 2)))

    def train(self):
        for _ in range(self.epochs):
            self.save()
            self.train_step()

    def train_step(self):
        a__: np.array = np.full(self.c, self.a)
        x_ = self.data[0]
        y_ = self.data[1]
        z_ = self.data[2]
        for x, y, z in zip(x_, y_, z_):
            b__: np.array = self.basis_function(x, y)
            g = self.gi(x, y)
            g__: np.array = np.full(self.c, g)
            z__: np.array = np.full(self.c, z)
            self.w -= a__ * (g__ - z__) * b__

    def save(self):
        self.error_history = np.append(self.error_history, self.e())