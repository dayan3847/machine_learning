class Model2D(ABC):
    def __init__(self, cxd: int = 5, epochs: int = 10, a: float = 0.1):
        self.a: float = a  # Learning rate
        self.cxd: int = cxd  # Number of basis functions for dimensions
        self.epochs: int = epochs  # Number of epochs

        # example shape(3,1000,)
        # 1000 cantidad de ejemplos
        # de los 3, el ultimo de Y y los anteriores forman el vector X
        self.data: np.array = np.loadtxt('data_3d.csv', delimiter=',').T  # Load Data
        self.n: int = data.shape[1]  # Examples count

        self.c: int = self.cxd ** 2  # Number of basis functions
        # Weights Ex: shape(25,)
        self.w__: np.array = np.random.rand(self.c) - .5  # Weights
        self.a__: np.array = np.full(self.c, self.a)  # Learning rate for each weight

        self.error_history: np.array = np.array([])  # Error history

        # Check shapes
        ShapeChecker.check_shape_data(self.data)
        ShapeChecker.check_shape(self.w__, (self.c,))
        ShapeChecker.check_shape(self.a__, (self.c,))

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
        return np.dot(self.w__, bf)

    # Calculate the model value for a vector
    def h(self, x: np.array, y: np.array) -> np.array:
        return np.array([self.hi(xi, yi) for xi, yi in zip(x, y)])

    def activate(self, h: float) -> float:
        return 1 / (1 + np.exp(-h))

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
        a = self.a
        x_ = self.data[0]
        y_ = self.data[1]
        z_ = self.data[2]
        for x, y, z in zip(x_, y_, z_):
            b = self.basis_function(x, y)
            g = self.gi(x, y)
            for i in range(self.c):
                self.w__[i] -= a * (g - z) * b[i]

    def save(self):
        self.error_history = np.append(self.error_history, self.e())
