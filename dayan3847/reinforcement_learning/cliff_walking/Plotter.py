from IPython.display import clear_output
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class Plotter:
    def __init__(
            self,
            queue_: mp.Queue,
            title_: str = '',
            fig_size: tuple = (15, 5)
    ):
        self.queue = queue_
        self.title = title_
        self.p_axs: list[PlotterAx] = []
        self.fig: Figure = plt.figure(figsize=fig_size)

    def get_ax(self, pos: int) -> plt.Axes:
        return self.fig.add_subplot(pos)

    def add_p_ax(self, p_ax_: 'PlotterAx'):
        self.p_axs.append(p_ax_)

    def plot(self, animate: bool = True, clear_output_: bool = False):
        if self.title != '':
            self.fig.suptitle(self.title)
        # for pax in self.p_axs:
        #     pax.plot()

        if animate:
            _ani = FuncAnimation(
                self.fig,
                self.plot_callback,
                blit=True,
                interval=17  # for 60 fps use interval=17
            )
        plt.tight_layout()
        if clear_output_:
            clear_output(wait=True)
        plt.show()

    def get_queue_data(self) -> dict | None:
        current_data = None
        while not self.queue.empty():
            current_data: np.array = self.queue.get()
        return current_data

    def plot_callback(self, frame) -> tuple:
        qdata = self.get_queue_data()
        r_ = tuple(pax.plot_callback(qdata) for pax in self.p_axs)
        self.fig.canvas.draw()
        return r_


class PlotterAx:

    def __init__(self, ax_: plt.Axes):
        self.ax: plt.Axes = ax_

    def plot(self):
        pass

    def plot_callback(self, qdata):
        pass


class PlotterAxLine2D(PlotterAx):
    def __init__(
            self,
            ax_: plt.Axes,
            title: str,
            label: str,
            xlabel: str,
            ylabel: str,
    ):
        super().__init__(ax_)
        self.title = title
        self.label = label

        # _current_data = self.get_queue_data()
        # self.ax.set_title(_current_data['title'])
        # self.data: Line2D = self.ax.plot(_current_data['x'], _current_data['y'], label=label, c='b')[0]
        self.data: Line2D = self.ax.plot([], [], label=label, c='b')[0]
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()

    def plot_callback(self, qdata):
        if qdata is not None:
            self.ax.set_title(qdata['title'])
            rewards_sum = qdata['rewards_sum']
            self.data.set_xdata(np.arange(len(rewards_sum)))
            self.data.set_ydata(rewards_sum / (qdata['experiments'][0] + 1))
            self.ax.relim()
            self.ax.autoscale_view()

        return self.data


class PlotterAxImg(PlotterAx):
    def __init__(
            self,
            ax_: plt.Axes,
            title: str = '',
            label: str = '',
    ):
        super().__init__(ax_)
        self.title = title
        self.label = label
        # self.ax.set_title(title)

        # _current_data = self.get_queue_data()
        # self.data = self.ax.imshow()
        # reiniciar la escala de los ejes
        self.ax.relim()

    def plot_callback(self, qdata):
        if qdata is not None:
            self.data.set_data(qdata['board'])

        return self.data


class PlotterAxMatrix(PlotterAx):
    def __init__(
            self,
            ax_: plt.Axes,
            title: str = '',
            label: str = '',
            xlabel: str = '',
            ylabel: str = '',
    ):
        super().__init__(ax_)
        self.title = title
        self.label = label
        self._plot()

        self.ax.set_title(title)
        # reiniciar la escala de los ejes
        self.ax.relim()

    def _plot(self):
        _current_data = self.get_queue_data()
        matrix = _current_data['data']
        m, n = matrix.shape
        self.data = self.ax.matshow(matrix, cmap='Greys')
        for i in range(m):
            for j in range(n):
                self.ax.text(j, i, str(matrix[i, j]), va='center', ha='center', color='red', fontsize=10)

    def plot_callback(self):
        self.data = self.ax.text(.2, .2, 'Heloo', va='center', ha='center', color='white')
        if not self.queue.empty():
            _current_data = self.get_queue_data()
            matrix = _current_data['data']
            m, n = matrix.shape
            self.data = self.ax.text(1, 1, 'Heloo', va='center', ha='center', color='white')
            # self.data = self.ax.matshow(matrix, cmap='viridis')
            # for i in range(m):
            #     for j in range(n):
            #         self.ax.text(j, i, str(matrix[i, j]), va='center', ha='center', color='white')

        return self.data
