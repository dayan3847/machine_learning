import numpy as np
from tensorflow import keras


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
        status_encoded_shape = (frames_count, *self.model_encoder.output_shape[1:])

        self.model_learning = self.build_model_learning(count_actions, status_encoded_shape)
        self.model_learning.compile(optimizer='adam', loss='mean_squared_error')

        # actions in one hot vector
        self.actions: np.array = np.eye(count_actions)
        self.path_h5 = 'knowledge/my_q_net.h5'

    # My Q NET
    @staticmethod
    def build_model_learning(count_actions, status_encoded_shape) -> keras.models.Model:
        status_input = keras.layers.Input(shape=status_encoded_shape, name='status_input')
        status_flatten = keras.layers.Flatten(name='status_flatten')(status_input)
        status_zip = keras.layers.Dense(20, name='status_zip')(status_flatten)
        action_input = keras.layers.Input(shape=(count_actions,), name='action_input')
        action_zip = keras.layers.Dense(20, name='action_zip')(action_input)
        action_status = keras.layers.Concatenate(name='concatenate_action_status')([status_zip, action_zip])
        general_layer_output = keras.layers.Dense(1, name='q')(action_status)

        return keras.models.Model(
            name='MY_Q_NET',
            inputs=[status_input, action_input],
            outputs=[general_layer_output],
        )

    def read_q_value(self,
                     a: int,  # action
                     s: np.array,  # status
                     ) -> float:
        _action = self.actions[a]

        # codificar el estado con el encoder (resnet50)
        _status_encoded = self.model_encoder.predict(s)

        # obtener el Q con la red neuronal usando el estado codificado y la accion
        _prediction = self.model_learning.predict([
            np.expand_dims(_status_encoded, axis=0),
            np.expand_dims(_action, axis=0)
        ])
        q = _prediction[0][0]
        return q

    def update_q_value(self,
                       a,  # action
                       s,  # status (normalmente seria el estado previo)
                       q,  # q_value
                       ):
        _action = self.actions[a]
        # codificar el estado con el encoder (resnet50)
        _status_encoded = self.model_encoder.predict(s)
        # entrenar la red neuronal con el estado codificado y la accion
        print('fit')
        self.model_learning.fit(
            x=[
                np.expand_dims(_status_encoded, axis=0),
                np.expand_dims(_action, axis=0)
            ],
            y=np.array([q]),
            epochs=1,
            verbose=0,
        )

    def save_knowledge(self):
        self.model_learning.save(self.path_h5)

    def load_knowledge(self):
        self.model_learning = keras.models.load_model(self.path_h5)


class Agent:
    def __init__(self,
                 env_,
                 frames: list[np.array],
                 state_frames_count: int = 4,
                 action_count: int = 11,
                 ):
        self.env = env_
        spec = env_.action_spec()
        self.action_count: int = action_count
        self.action_values: np.array = np.linspace(spec.minimum, spec.maximum, action_count)

        # q-learning
        self.alpha = .1
        self.gamma = 1
        self.epsilon = .1

        self.state_frames_count: int = state_frames_count
        if len(frames) < state_frames_count:
            raise Exception('Frames count is less than state_frames_count')
        self.frames: list[np.array] = frames
        self.state = self.update_status()

        self.knowledge_model = KnowledgeModel(
            count_actions=action_count,
            frames_shape=frames[0].shape,
            frames_count=self.state_frames_count,
        )
        # Cantidad de acciones que se han realizado
        self.step_count = 0

    def update_status(self):
        self.state = np.array(self.frames[-self.state_frames_count:])
        return self.state

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
            s = self.state
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

    def run_step(self) -> float:
        a: int = self.select_an_action()
        state_prev = self.state
        reward = self.apply_action(a)
        self.train_action(a, state_prev, reward)
        self.step_count += 1
        # Guardar el conocimiento cada 100 pasos
        if self.step_count % 10 == 0:
            print('saving knowledge')
            self.knowledge_model.save_knowledge()
        return reward

    def apply_action(self, a: int):
        action_value_: float = self.action_values[a]
        time_step_ = self.env.step(action_value_)

        print("Position: ", time_step_.observation['position'])
        print("Velocity: ", time_step_.observation['velocity'])
        _r = time_step_.reward
        print("Reward: ", _r)
        # Get new Frame
        _f = self.env.physics.render(camera_id=0)
        self.frames.append(_f)
        self.update_status()
        return _r

    def train_action(self, a: int, state_prev: np.array, reward: float):
        _q = self.knowledge_model.read_q_value(a, state_prev)
        _q_as_max = self.select_an_action_best_q()[1]
        _q_fixed: float = _q + self.alpha * (reward + self.gamma * _q_as_max - _q)
        self.knowledge_model.update_q_value(a, state_prev, _q_fixed)
