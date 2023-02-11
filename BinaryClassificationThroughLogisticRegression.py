import matplotlib.pyplot as plt


class BinaryClassificationThroughLogisticRegression:

    def __init__(self):
        self.data_points = None

    def init(self):
        self.load_data_points()

    # Generate Data Points
    def load_data_points(self):
        if self.data_points is not None:
            return
        file = open('data.txt', 'r')
        for line in file:


        file.close()
        x1_list = []
        x2_list = []
        y_list = []
        for line in lines:
            line = line.strip()
            (x1, x2, y) = line.split()
            x1_list.append(float(x1))
            x2_list.append(float(x2))
            y_list.append(int(y))
        self.data_points = {
            'x_list': [],
            'y_list': [],
            'z_list': [],
        }

    def plot_data(self):
        # Data
        plt.clf()
        plt.title('Data')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.axvline(color='black')
        plt.axhline(color='black')
        # points
        true_x1 = self.data_points[True]['x_list']
        true_x2 = self.data_points[True]['y_list']
        false_x1 = self.data_points[False]['x_list']
        false_x2 = self.data_points[False]['y_list']
        plt.scatter(true_x1, true_x2, color='green', label='positive')
        plt.scatter(false_x1, false_x2, color='red', label='negative')
        plt.legend()
        plt.grid()
        plt.show()

    def main(self, plot: bool = False):
        self.init()
        self.plot_data()


if __name__ == '__main__':
    controller = BinaryClassificationThroughLogisticRegression()
    controller.main()
