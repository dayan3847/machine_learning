import time
import numpy as np
import multiprocessing as mp

from Plotter import Plotter

np.random.seed(0)


class Environment:
    def __init__(self):
        self.board_shape: tuple[int, int] = (4, 12)
        # Add board
        self.board: np.array = np.full(self.board_shape, -1)
        # Add cliff
        self.board[-1, 1:-1] = -100
        # Add goal
        self.board[-1, -1] = 100
        self.actions: list[np.array] = [
            np.array([-1, 0]),  # up
            np.array([1, 0]),  # down
            np.array([0, -1]),  # left
            np.array([0, 1]),  # right
        ]
        self.init_state: np.array = np.array([3, 0])

    # Get actions available by state
    def get_actions_available(self, state: np.array) -> np.ndarray:
        actions = [0, 1, 2, 3]
        if state[0] == 0:
            actions.remove(0)
        elif state[0] == self.board_shape[0] - 1:
            actions.remove(1)
        if state[1] == 0:
            actions.remove(2)
        elif state[1] == self.board_shape[1] - 1:
            actions.remove(3)
        return np.array(actions)

    def apply_action(self, ag: 'Agent', action: int) -> (float, bool):
        episode_end: bool = False
        # Validate action
        actions_available = self.get_actions_available(ag.state)
        if action not in actions_available:
            raise Exception('Invalid action')
        # Apply action
        ag.state = ag.state + self.actions[action]
        # Get reward
        try:
            reward: float = self.board[tuple(ag.state)]
        except IndexError:
            print(ag.state)
            raise Exception('Invalid position')
        # Validate position
        if abs(reward) == 100:
            ag.state = self.init_state
            episode_end = True

        return reward, episode_end


class Agent:
    def __init__(self, env_: Environment):
        self.env: Environment = env_
        self.state: np.array = self.env.init_state
        self.running: bool = False

    def decide_an_action(self, actions: np.array) -> int:
        pass

    def train_action(self, action: int, state: np.array, reward: float):
        pass

    def run_step(self):
        actions: np.array = self.env.get_actions_available(self.state)
        if len(actions) == 0:
            raise Exception('No actions available')
        action: int = self.decide_an_action(actions)
        before_state = self.state
        reward, episode_end = self.env.apply_action(self, action)
        self.train_action(action, before_state, reward)
        return reward, episode_end

    @staticmethod
    def decide_an_action_random(actions: np.array) -> int:
        index: int = np.random.randint(0, len(actions))
        return actions[index]


class AgentRandom(Agent):
    def decide_an_action(self, actions: np.array) -> int:
        return self.decide_an_action_random(actions)


class AgentQLearning(Agent):
    def __init__(self, env_: Environment):
        super().__init__(env_)
        self.Q = self.init_q()
        self.alpha = .1
        self.gamma = 1
        self.epsilon = .1

    def init_q(self):
        self.Q = np.zeros((4, 4, 12))
        return self.Q

    def decide_an_action(self, actions: np.array) -> int:
        _action: int = self.decide_an_action_random(actions) if np.random.random() < self.epsilon \
            else self.decide_an_action_best_q(actions)[0]
        return _action

    # Obtener el valor de Q para una accion
    def get_q_value(self, action: int, state=None) -> float:
        if state is None:
            state = self.state
        return float(self.Q[action, state[0], state[1]])

    # Obtener los valores de Q para varias acciones
    # result action -> q_value
    def get_q_values(self, actions: np.array, state=None) -> np.array:
        _r = [[action, self.get_q_value(action, state)] for action in actions]
        return np.array(_r)

    def decide_an_action_best_q(self, actions: np.array, state=None) -> (int, float):
        best_actions = np.array([])
        best_q_value = -np.inf
        q_values_per_action = self.get_q_values(actions, state)
        for av in q_values_per_action:
            a = av[0]
            v = av[1]
            if v > best_q_value:
                best_q_value = v
                best_actions = np.array([a])
            elif v == best_q_value:
                best_actions = np.append(best_actions, a)
        if 0 == len(best_actions):
            raise Exception('Best action not found')
        # TODO la idea es que se elija una accion aleatoria entre las mejores
        # best_action = np.random.choice(best_actions)
        # TODO para estas pruebas se elige la primera accion
        best_action = best_actions[0]
        return int(best_action), float(best_q_value)

    def train_action(self, action: int, state: np.array, reward: float):
        # _q = self.get_q_value(action, state)
        _q = self.Q[action, state[0], state[1]]
        _q_as_max = self.decide_an_action_best_q(self.env.get_actions_available(self.state), self.state)[1]
        _q_fixed: float = _q + self.alpha * (reward + self.gamma * _q_as_max - _q)
        self.Q[action, state[0], state[1]] = _q_fixed


class Trainner:
    def __init__(self):
        self.experiments_status: tuple[int, int] = 0, 50  # Experiment 0 of 1
        self.episodes_status: tuple[int, int] = 0, 500  # Episode 0 of 500
        self.env: Environment = Environment()
        self.agent: AgentQLearning = AgentQLearning(self.env)
        # rewards promedio(de todos los experimentos) por episodio
        self.rewards_sum: np.array = np.zeros(self.episodes_status[1])
        self.success: np.array = np.zeros(self.episodes_status[1])

    @staticmethod
    def int_to_color(value: int) -> tuple[int, int, int]:
        if value == -1:
            return (192, 192, 192)
        elif value == -100:
            return (128, 128, 128)
        elif value == 100:
            return (0, 255, 0)
        else:
            return (0, 0, 0)

    def get_board_color(self):
        _r = np.array([[self.int_to_color(val) for val in row] for row in self.env.board])
        # init state
        _r[tuple(self.env.init_state)] = (128, 128, 255)
        # agent state
        _r[tuple(self.agent.state)] = (255, 255, 0)
        return _r

    def get_title(self):
        return 'Experiment: {}/{} Episode: {}/{}'.format(
            self.experiments_status[0] + 1,
            self.experiments_status[1],
            self.episodes_status[0] + 1,
            self.episodes_status[1]
        )

    def get_status(self) -> dict:
        q = self.agent.Q
        # redondear a 2 decimales
        q = np.round(q, 2)

        q_best = np.ndarray((4, 12), dtype=int)
        for i in range(4):
            for j in range(12):
                state = np.array([i, j])
                actions: np.array = self.env.get_actions_available(state)
                a = self.agent.decide_an_action_best_q(actions, state)
                q_best[i, j] = a[0]

        return {
            'title': self.get_title(),
            'experiments': self.experiments_status,
            'episodes': self.episodes_status,
            'board': self.get_board_color(),
            'rewards_sum': self.rewards_sum,
            'q': {
                'up': q[0],
                'down': q[1],
                'left': q[2],
                'right': q[3],
                'best': q_best,
            }
        }

    def train_callback(
            self,
            queue_status_: mp.Queue,
            queue_stop_: mp.Queue,
    ):
        for i in range(self.experiments_status[1]):
            self.experiments_status = i, self.experiments_status[1]
            self.agent = AgentQLearning(self.env)
            # agent: Agent = AgentRandom(env)
            for j in range(self.episodes_status[1]):
                self.episodes_status = j, self.episodes_status[1]
                print(self.get_title())
                episode_end: bool = False
                while not episode_end:
                    # time.sleep(1)
                    reward, episode_end = self.agent.run_step()
                    self.rewards_sum[j] += reward
                    if episode_end and reward > 0:
                        self.success[j] += 1
                    # Enqueue status
                    queue_status_.put(self.get_status())
            if not queue_stop_.empty():
                queue_stop_.get()
                print('stop')
                return


if __name__ == '__main__':
    queue_status: mp.Queue = mp.Queue()
    queue_stop: mp.Queue = mp.Queue()
    t = Trainner()
    queue_status.put(t.get_status())
    process: mp.Process = mp.Process(
        target=t.train_callback,
        args=(
            queue_status,
            queue_stop,
        ),
    )
    time.sleep(1)
    process.start()
    p: Plotter = Plotter(queue_status)
    p.plot()
    queue_stop.put(1)
    process.join()
