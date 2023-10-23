from IPython.display import clear_output
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure


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

        qdata = self.get_queue_data()

        _ax = self.get_ax(241)
        self.add_p_ax(PlotterAxImg(_ax, qdata))

        _ax = self.get_ax(242)
        self.add_p_ax(PlotterAxMatrix(_ax, qdata, title='best'))

        _ax = self.get_ax(244)
        self.add_p_ax(PlotterAxLine2D(_ax, qdata,
                                      label='Rewards',
                                      xlabel='Episode',
                                      ylabel='Sum of rewards'
                                      ))

        _ax = self.get_ax(243)
        self.add_p_ax(PlotterAxMatrix(_ax, qdata, title='up'))
        _ax = self.get_ax(247)
        self.add_p_ax(PlotterAxMatrix(_ax, qdata, title='down'))
        _ax = self.get_ax(246)
        self.add_p_ax(PlotterAxMatrix(_ax, qdata, title='left'))
        _ax = self.get_ax(248)
        self.add_p_ax(PlotterAxMatrix(_ax, qdata, title='right'))

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
        r_ = []
        for pax in self.p_axs:
            r_ax = pax.plot_callback(qdata)
            r_.extend(r_ax)
        r_ = tuple(r_)
        self.fig.canvas.draw()
        return r_


class PlotterAx:

    def __init__(self, ax_: plt.Axes):
        self.ax: plt.Axes = ax_

    def plot(self):
        pass

    def plot_callback(self, qdata) -> list:
        pass


class PlotterAxLine2D(PlotterAx):
    def __init__(
            self,
            ax_: plt.Axes,
            qdata: dict,
            label: str,
            xlabel: str,
            ylabel: str,
    ):
        super().__init__(ax_)
        self.label = label

        self.ax.set_title(qdata['title'])
        rewards_sum = qdata['rewards_sum']
        self.data = self.ax.plot(
            np.arange(len(rewards_sum)),
            rewards_sum / (qdata['experiments'][0] + 1),
            label=label, c='b')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.legend()

    def plot_callback(self, qdata) -> list:
        if qdata is not None:
            self.ax.set_title(qdata['title'])
            rewards = qdata['rewards']
            self.data[0].set_xdata(np.arange(len(rewards)))
            self.data[0].set_ydata(rewards)
            print(f"denominador: {qdata['experiments'][0] + 1}")
            self.ax.relim()
            self.ax.autoscale_view()

        return self.data


class PlotterAxImg(PlotterAx):
    def __init__(
            self,
            ax_: plt.Axes,
            qdata: dict,
    ):
        super().__init__(ax_)
        self.ax.set_title(qdata['title'])
        self.data = self.ax.imshow(qdata['board'])
        self.ax.relim()

    def plot_callback(self, qdata) -> list:
        if qdata is not None:
            self.ax.set_title(qdata['title'])
            self.data.set_data(qdata['board'])

        return [self.data]


class PlotterAxMatrix(PlotterAx):
    def __init__(
            self,
            ax_: plt.Axes,
            qdata: dict,
            title: str,
    ):
        super().__init__(ax_)
        self.title = title
        self._plot(qdata)

        self.ax.set_title(title)

        # reiniciar la escala de los ejes
        self.ax.relim()

    def _plot(self, qdata: dict):
        self.data = []
        matrix = qdata['q'][self.title]
        m, n = matrix.shape
        self.data = [self.ax.matshow(matrix, cmap='Greys')]
        for i in range(m):
            for j in range(n):
                cell = str(matrix[i, j])
                if self.title == 'best':
                    cell = '↑' if cell == '0' else cell
                    cell = '↓' if cell == '1' else cell
                    cell = '←' if cell == '2' else cell
                    cell = '→' if cell == '3' else cell
                _t = self.ax.text(j, i, cell, va='center', ha='center', color='red', fontsize=10)
                self.data.append(_t)

    def plot_callback(self, qdata) -> list:
        if qdata is not None:
            self._plot(qdata)

        return self.data
