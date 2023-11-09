import numpy as np
from tensorflow import keras
from dm_control.rl.control import Environment
from dm_env import StepType, TimeStep


class KnowledgeModel:
    def __init__(self,
                 count_actions: int,  # cantidad de acciones
                 frames_shape: tuple,  # shape de los frames
                 frames_count: int,  # cantidad de frames que va a tener el estado
                 ):
        self.model_encoder = keras.applications.resnet50.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=frames_shape,
        )
        state_encoded_shape = (frames_count, *self.model_encoder.output_shape[1:])

        self.model_learning = self.build_model_learning(count_actions, state_encoded_shape)
        self.model_learning.compile(optimizer='adam', loss='mean_squared_error')

        # actions in one hot vector
        self.actions: np.array = np.eye(count_actions)
        self.path_h5 = 'knowledge/my_q_net.h5'

    # My Q NET
    @staticmethod
    def build_model_learning(count_actions, state_encoded_shape) -> keras.models.Model:
        state_input = keras.layers.Input(shape=state_encoded_shape, name='state_input')
        state_flatten = keras.layers.Flatten(name='state_flatten')(state_input)
        state_zip = keras.layers.Dense(5, name='state_zip')(state_flatten)
        action_input = keras.layers.Input(shape=(count_actions,), name='action_input')
        action_zip = keras.layers.Dense(5, name='action_zip')(action_input)
        action_state = keras.layers.Concatenate(name='concatenate_action_state')([state_zip, action_zip])
        general_layer_output = keras.layers.Dense(1, name='q')(action_state)

        return keras.models.Model(
            name='MY_Q_NET',
            inputs=[state_input, action_input],
            outputs=[general_layer_output],
        )

    def read_q_value(self,
                     a: int,  # action
                     s: np.array,  # state
                     ) -> float:
        _action = self.actions[a]

        # codificar el estado con el encoder (resnet50)
        _state_encoded = self.model_encoder.predict(s)

        # obtener el Q con la red neuronal usando el estado codificado y la accion
        _prediction = self.model_learning.predict([
            np.expand_dims(_state_encoded, axis=0),
            np.expand_dims(_action, axis=0)
        ])
        q = _prediction[0][0]
        return q

    def update_q_value(self,
                       a,  # action
                       s,  # state (normalmente seria el estado previo)
                       q,  # q_value
                       ):
        _action = self.actions[a]
        # codificar el estado con el encoder (resnet50)
        _state_encoded = self.model_encoder.predict(s)
        # entrenar la red neuronal con el estado codificado y la accion
        print('fit')
        self.model_learning.fit(
            x=[
                np.expand_dims(_state_encoded, axis=0),
                np.expand_dims(_action, axis=0)
            ],
            y=np.array([q]),
            epochs=1,
            verbose=0,
        )

    def save_knowledge(self):
        self.model_learning.save(f'{self.path_h5}_')
        keras.models.save_model(self.model_learning, self.path_h5)

    def load_knowledge(self):
        self.model_learning = keras.models.load_model(self.path_h5)


class Agent:
    def __init__(self,
                 env: Environment,
                 action_count: int,
                 state_frames_count: int,
                 frames_shape: tuple[int, int, int],
                 ):
        self.env: Environment = env
        spec = env.action_spec()
        self.action_count: int = action_count
        self.action_values: np.array = np.linspace(spec.minimum, spec.maximum, action_count)

        # q-learning
        self.alpha = .1
        self.gamma = 1
        self.epsilon = .1

        self.state_frames_count: int = state_frames_count
        self.frames: list[np.array] = []

        self.frames_shape: tuple[int, int, int] = frames_shape
        self.knowledge_model = KnowledgeModel(
            count_actions=action_count,
            frames_shape=self.frames_shape,
            frames_count=self.state_frames_count,
        )

    def get_state_current(self) -> np.array:
        return np.array(self.frames[-self.state_frames_count:])

    def get_state_prev(self) -> np.array:
        return np.array(self.frames[-self.state_frames_count - 1:-1])

    def select_an_action(self) -> int:
        # Realizara una accion aleatoria con probabilidad epsilon
        return self.select_an_action_random() if np.random.random() < self.epsilon \
            else self.select_an_action_best_q()[0]

    def select_an_action_random(self) -> int:
        return np.random.randint(self.action_count)

    def select_an_action_best_q(self,
                                s=None,  # state (default self.state)
                                ) -> tuple[int, float]:  # best_action, best_q_value
        if s is None:
            s = self.get_state_current()
        # Obtener la lista de valores Q de todas las acciones para el estado "s"
        q_values_per_action = self.read_q_values_x_actions(s)
        # De la lista de valores Q, buscar el mejor
        best_action = q_values_per_action[0][0]
        best_q_value = q_values_per_action[0][1]
        for av in q_values_per_action[1:]:
            a = av[0]  # action
            v = av[1]  # q_value
            if v > best_q_value:
                best_action = a
                best_q_value = v
        return int(best_action), float(best_q_value)

    def read_q_values_x_actions(self, s: np.array) -> list:
        return [(a, self.knowledge_model.read_q_value(a, s)) for a in range(self.action_count)]

    def run_step(self) -> TimeStep:
        a: int = self.select_an_action()
        time_step = self.apply_action(a)
        self.train_action(a, time_step.reward)
        return time_step

    def apply_action(self, a: int) -> TimeStep:
        action_value_: float = float(self.action_values[a])
        time_step = self.env.step(action_value_)
        self.frames.append(self.get_current_frame())
        return time_step

    def get_current_frame(self):
        return self.env.physics.render(camera_id=0, height=self.frames_shape[0], width=self.frames_shape[1])

    def train_action(self, a: int, reward: float):
        state_prev: np.array = self.get_state_prev()
        _q = self.knowledge_model.read_q_value(a, state_prev)
        _q_as_max = self.select_an_action_best_q()[1]
        _q_fixed: float = _q + self.alpha * (reward + self.gamma * _q_as_max - _q)
        self.knowledge_model.update_q_value(a, state_prev, _q_fixed)

    def run_episode(self):
        reward: list[float] = []
        time_step = self.env.reset()
        _f = self.get_current_frame()
        self.frames = [_f for _ in range(self.state_frames_count)]
        step: int = 0
        while StepType.LAST != time_step.step_type:
            step += 1
            print("\033[92m{}\033[00m".format(step))
            time_step = self.run_step()
            _r: float = float(time_step.reward)
            reward.append(_r)
            print("Reward: ", _r)
            # print("Position: ", time_step.observation['position'])
            # print("Velocity: ", time_step.observation['velocity'])

        print('saving knowledge')
        self.knowledge_model.save_knowledge()
        print('saving reward')
        np.savetxt('knowledge/reward.txt', reward)
        print('saving frames')
        np.save('knowledge/frames.npy', self.frames)
