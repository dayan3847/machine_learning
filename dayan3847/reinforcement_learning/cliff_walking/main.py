import time
import numpy as np
import multiprocessing as mp

from Plotter import Plotter, PlotterAx


class Environment:
    def __init__(self):
        self.limit: np.array = np.array([4, 12])
        self.board: np.array = np.array([
            np.full(12, -1),
            np.full(12, -1),
            np.full(12, -1),
            np.array([-1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 100]),
        ])
        self.actions: list[np.array] = [
            np.array([-1, 0]),  # up
            np.array([1, 0]),  # down
            np.array([0, -1]),  # left
            np.array([0, 1]),  # right
        ]
        self.init_pos: np.array = np.array([3, 0])

    def get_actions_available(self, ag: 'Agent') -> np.ndarray:
        actions = [0, 1, 2, 3]
        if ag.pos[0] == 0:
            actions.remove(0)
        elif ag.pos[0] == self.limit[0] - 1:
            actions.remove(1)
        if ag.pos[1] == 0:
            actions.remove(2)
        elif ag.pos[1] == self.limit[1] - 1:
            actions.remove(3)
        return np.array(actions)

    def apply_action(self, ag: 'Agent', action: int) -> (float, bool):
        episode_end: bool = False
        # Validate action
        if action not in self.get_actions_available(ag):
            raise Exception('Invalid action')
        # Apply action
        ag.pos = ag.pos + self.actions[action]
        # Get reward
        try:
            reward: float = self.board[tuple(ag.pos)]
        except IndexError:
            print(ag.pos)
            raise Exception('Invalid position')
        # Validate position
        if abs(reward) == 100:
            ag.pos = self.init_pos
            episode_end = True

        return reward, episode_end


class Agent:
    def __init__(self, env_: Environment):
        self.env: Environment = env_
        self.pos: np.array = self.env.init_pos
        self.running: bool = False

    def get_action(self) -> int:
        pass

    def train_action(self, reward: float):
        pass

    def run_step(self):
        action: int = self.get_action()
        reward, episode_end = self.env.apply_action(self, action)
        if not episode_end:
            self.train_action(reward)
        return reward, episode_end


class AgentRandom(Agent):
    def get_action(self) -> int:
        actions: np.array = self.env.get_actions_available(self)
        index: int = np.random.randint(0, len(actions))
        return actions[index]


class Trainner:
    def __init__(self):
        self.experiments_count: int = 100
        self.episodes_count: int = 100
        self.rewards: np.array = np.zeros(self.episodes_count)
        env: Environment = Environment()
        self.agent: Agent = AgentRandom(env)

    def train_callback(self, queue_rewards_: mp.Queue, queue_stop_: mp.Queue):
        for i in range(self.experiments_count):
            for j in range(self.episodes_count):
                title = 'Experiment: {}/{} Episode: {}/{}'.format(
                    i + 1,
                    self.experiments_count,
                    j + 1,
                    self.episodes_count
                )
                while True:
                    reward, episode_end = self.agent.run_step()
                    self.rewards[j] += reward
                    current_data = {
                        'title': title,
                        'x': np.arange(self.episodes_count),
                        'y': self.rewards,
                    }
                    queue_rewards_.put(current_data)
                    if episode_end:
                        break
                    if not queue_stop_.empty():
                        queue_stop_.get()
                        print('stop')
                        return


def get_plot(
        queue_rewards_: mp.Queue,
):
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
