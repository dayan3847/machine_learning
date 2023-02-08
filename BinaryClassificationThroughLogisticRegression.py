import matplotlib.pyplot as plt


class BinaryClassificationThroughLogisticRegression:

    def __init__(self):
        self.data_point = None

    def init(self):
        self.load_data_points()

    # Generate Data Points
    def load_data_points(self):
        if self.data_point is not None:
            return
        file = open('data.txt', 'r')
        lines = file.readlines()
        file.close()
        x1_list = []
        x2_list = []
        y_list = []
        for line in lines:
            line = line.strip()
            (x1, x2, y) = line.split()
            x1_list.append(float(x1))
            x2_list.append(float(x2))
            y_list.append(1 == int(y))
        self.data_point = {
            True: {
                'x_list': [],
                'y_list': [],
            },
            False: {
                'x_list': [],
                'y_list': [],
            },
        }
        for i in range(len(x1_list)):
            x1i = x1_list[i]
            x2i = x2_list[i]
            yi = y_list[i]
            self.data_point[yi]['x_list'].append(x1i)
            self.data_point[yi]['y_list'].append(x2i)

    def plot_data(self):
        # Data
        plt.clf()
        plt.title('Data')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.axvline(color='black')
        plt.axhline(color='black')
        # points
        true_x1 = self.data_point[True]['x_list']
        true_x2 = self.data_point[True]['y_list']
        false_x1 = self.data_point[False]['x_list']
        false_x2 = self.data_point[False]['y_list']
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
