import numpy as np
from dayan3847.tools import ModelGaussian


class Environment:
    def __init__(self):
        # BOARD
        self.board_shape: tuple[int, int] = (4, 12)
        # Add board
        self.board_reward: np.array = np.full(self.board_shape, -1)
        # Add cliff
        self.board_reward[-1, 1:-1] = -100
        # Add goal
        self.board_reward[-1, -1] = 100

        self.board_incidence: np.array = np.zeros(self.board_shape)
        # ACTIONS
        self.actions: list[np.array] = [
            np.array([-1, 0]),  # up
            np.array([1, 0]),  # down
            np.array([0, -1]),  # left
            np.array([0, 1]),  # right
        ]
        self.actions_count: int = len(self.actions)
        # CONFIG
        self.init_state: np.array = np.array([3, 0])
        # STATS
        self.count_win: int = 0
        self.count_lose: int = 0

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
            reward: float = self.board_reward[tuple(ag.state)]
        except IndexError:
            print(ag.state)
            raise Exception('Invalid position')
        self.board_incidence[tuple(ag.state)] += 1
        # Validate position
        if abs(reward) == 100:
            ag.state = self.init_state
            episode_end = True
            if reward > 0:
                self.count_win += 1
            else:
                self.count_lose += 1

        return reward, episode_end


class Agent:
    def __init__(self, env_: Environment):
        self.env: Environment = env_
        self.state: np.array = self.env.init_state

    def run_step(self):
        actions: np.array = self.env.get_actions_available(self.state)
        if len(actions) == 0:
            raise Exception('No actions available')
        action: int = self.decide_an_action(actions)
        state_prev = self.state
        reward, episode_end = self.env.apply_action(self, action)
        self.train_action(action, state_prev, reward)
        return reward, episode_end

    @staticmethod
    def decide_an_action_random(actions: np.array) -> int:
        index: int = np.random.randint(0, len(actions))
        return actions[index]

    def decide_an_action(self, actions: np.array) -> int:
        pass

    def train_action(self, action: int, state_prev: np.array, reward: float):
        pass


class AgentRandom(Agent):
    def decide_an_action(self, actions: np.array) -> int:
        return self.decide_an_action_random(actions)


# Esta es una clase abstracta
class AgentQLearning(Agent):
    def __init__(self, env_: Environment):
        super().__init__(env_)
        self.alpha = .1
        self.gamma = 1
        self.epsilon = .1

    def decide_an_action(self, actions: np.array) -> int:
        _action: int = self.decide_an_action_random(actions) if np.random.random() < self.epsilon \
            else self.decide_an_action_best_q(actions)[0]
        return _action

    def decide_an_action_best_q(self, actions: np.array, state=None) -> (int, float):
        if state is None:
            state = self.state
        best_actions = np.array([])
        best_q_value = -np.inf
        q_values_per_action = self.read_q_values_x_actions(actions, state)
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

    # Obtener los valores de Q para varias acciones
    # result action -> q_value
    def read_q_values_x_actions(self, actions: np.array, state) -> list:
        _r = [(a, self.read_q_value(a, state)) for a in actions]
        return _r

    def train_action(self, action: int, state_prev: np.array, reward: float):
        _q = self.read_q_value(action, state_prev)
        _q_as_max = self.decide_an_action_best_q(self.env.get_actions_available(self.state), self.state)[1]
        _q_fixed: float = _q + self.alpha * (reward + self.gamma * _q_as_max - _q)
        self.update_q_value(action, state_prev, _q_fixed)

    def get_best_actions_for_all_states(self) -> np.array:
        _r = np.full(self.env.board_shape, -1)
        for i in range(self.env.board_shape[0]):
            for j in range(self.env.board_shape[1]):
                state = np.array([i, j])
                actions: np.array = self.env.get_actions_available(state)
                if len(actions) == 0:
                    continue
                a = self.decide_an_action_best_q(actions, state)
                _r[i, j] = a[0]

        return _r

    # Leer para una accion y un estado el valor de Q
    def read_q_value(self, action: int, state=None) -> float:
        pass

    # Actualizar para una accion y un estado el valor de Q
    def update_q_value(self, action: int, state: np.array, new_value: float):
        pass

    def reset_knowledge(self):
        pass

    def save_knowledge(self):
        pass

    def load_knowledge(self):
        pass


class AgentQLearningTable(AgentQLearning):
    def __init__(self, env_: Environment):
        super().__init__(env_)
        self.q_table_models: list[np.array] = self.reset_knowledge()

    def reset_knowledge(self):
        self.q_table_models = [np.zeros(self.env.board_shape) for _ in range(self.env.actions_count)]
        return self.q_table_models

    def save_knowledge(self):
        for i in range(self.env.actions_count):
            np.savetxt('knowledge/q_table_{}.csv'.format(i), self.q_table_models[i], delimiter=',')

    def load_knowledge(self):
        for i in range(self.env.actions_count):
            self.q_table_models[i] = np.loadtxt('knowledge/q_table_{}.csv'.format(i), delimiter=',')

    # Leer para una accion y un estado el valor de Q
    def read_q_value(self, action: int, state=None) -> float:
        if state is None:
            state = self.state
        _t = self.q_table_models[action]
        _r = _t[state[0], state[1]]
        return float(_r)

    # Actualizar para una accion y un estado el valor de Q
    def update_q_value(self, action: int, state: np.array, new_value: float):
        self.q_table_models[action][state[0], state[1]] = new_value


class AgentQLearningGaussian(AgentQLearning):
    def __init__(self, env_: Environment, a: float = .1, s2: float = .1, init_weights_random: bool = True):
        super().__init__(env_)
        self.a = a  # learning rate
        self.s2 = s2  # variance ^ 2
        self.init_weights_random = init_weights_random
        self.q_gaussian_models: list[ModelGaussian] = self.reset_knowledge()

    def reset_knowledge(self):
        self.q_gaussian_models = [
            ModelGaussian(
                self.a,
                (4, 12),
                ((0, 4), (0, 12)),
                self.s2,
                self.init_weights_random,
            ) for _ in range(self.env.actions_count)
        ]
        return self.q_gaussian_models

    def save_knowledge(self):
        for i in range(self.env.actions_count):
            np.savetxt('knowledge/q_gaussian_{}.csv'.format(i), self.q_gaussian_models[i].weights_vfr.T, delimiter=',')

    def load_knowledge(self):
        for i in range(self.env.actions_count):
            self.q_gaussian_models[i].weights_vfr = np.loadtxt('knowledge/q_gaussian_{}.csv'.format(i), delimiter=',').T

    # Obtener el valor de Q para una accion
    # Leer para una accion y un estado el valor de Q
    def read_q_value(self, action: int, state=None) -> float:
        if state is None:
            state = self.state
        _m: ModelGaussian = self.q_gaussian_models[action]
        _r = _m.gi(state)
        return float(_r)

    # Actualizar para una accion y un estado el valor de Q
    def update_q_value(self, action: int, state: np.array, new_value: float):
        self.q_gaussian_models[action].update_w(state, new_value)
