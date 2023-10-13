import time
import numpy as np
import multiprocessing as mp

from Plotter import Plotter, PlotterAx


class Trainner:
    def __init__(self):
        self.experiments_count: int = 1000
        self.episodes_count: int = 100
        self.rewards: np.ndarray = np.zeros((self.experiments_count, self.episodes_count))

    def train_callback(self, queue_rewards_: mp.Queue, queue_stop_: mp.Queue):
        for i in range(self.experiments_count):
            for j in range(self.episodes_count):
                title = 'Experiment: {}/{} Episode: {}/{}'.format(
                    i + 1,
                    self.experiments_count,
                    j + 1,
                    self.episodes_count
                )
                self.rewards[i, j] = np.random.normal()
                sum_rewards = self.rewards.sum(axis=0)
                current_data = {
                    'title': title,
                    'x': np.arange(self.episodes_count),
                    'y': sum_rewards,
                }
                queue_rewards_.put(current_data)
                if not queue_stop_.empty():
                    print('stop')
                    return ''


def get_plot(queue_rewards_: mp.Queue):
    _p: Plotter = Plotter()
    _p.add_p_ax(PlotterAx(_p.get_ax(111), queue_rewards_))
    return _p


if __name__ == '__main__':
    queue_rewards: mp.Queue = mp.Queue()
    queue_stop: mp.Queue = mp.Queue()
    t = Trainner()

    process: mp.Process = mp.Process(
        target=t.train_callback,
        args=(queue_rewards, queue_stop),
    )
    time.sleep(1)
    process.start()
    p: Plotter = get_plot(queue_rewards)
    p.plot()
    queue_stop.put(1)
    process.join()
