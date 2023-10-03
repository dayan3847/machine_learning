import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from dayan3847.reinforcement_learning.case_1d import Agent


class Environment:
    MAX: np.array = np.array([10, 1])
    TIME_STEP: float = .3

    def __init__(self):
        self.agents: list[Agent] = []
        # Plot
        self.fig, self.ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
        self.a_ql: Agent | None = None
        self.q_actions_bars: np.ndarray = np.empty((3,), dtype=object)
        self.q = None

    def run(self):
        for agent in self.agents:
            agent.run()
        self.plot()
        self.stop()

    def stop(self):
        for agent in self.agents:
            agent.stop()

    def get_agent_by_name(self, name: str):
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def plot(self):
        self.a_ql = self.get_agent_by_name('q_learning')
        if self.a_ql is None:
            return
        statuses: np.array = np.arange(self.a_ql.Q.shape[0])
        for i in range(self.a_ql.Q.shape[1]):
            values: np.array = self.abs(self.a_ql.Q[:, i])
            self.q_actions_bars[i] = self.ax[i].bar(statuses, values, label='Status')
            self.ax[i].set_ylim(0.8, 1)

        self.ax[0].set_title('Left')
        self.ax[1].set_title('No Move')
        self.ax[2].set_title('Right')

        self.q = self.ax[3].imshow(self.a_ql.Q.T, cmap='viridis', interpolation='nearest')
        self.ax[3].set_title('Q')
        # colorbar
        #self.fig.colorbar(self.ax[3].get_images()[0], ax=self.ax[3])

        # create animation using the animate() function
        _ani = animation.FuncAnimation(self.fig, self.plot_callback, interval=Environment.TIME_STEP * 1000)

        plt.tight_layout()
        plt.show()

    def plot_callback(self, frame):
        for i in range(self.a_ql.Q.shape[1]):
            values: np.array = self.abs(self.a_ql.Q[:, i])
            for bar, v in zip(self.q_actions_bars[i], values):
                bar.set_height(v)
        # self.q.set_data(self.a_ql.Q.T)
        self.ax[3].imshow(self.a_ql.Q.T, cmap='viridis', interpolation='nearest')

    @staticmethod
    def abs(values):
        values = np.abs(values)
        values = values / np.max(values)
        # values = values / np.sum(values)
        return values
