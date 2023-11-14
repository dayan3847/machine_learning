import numpy as np
from dm_control.rl.control import Environment
from tensorflow import keras

from dayan3847.reinforcement_learning.deep_mind.QLearningAgent import QLearningAgent, KnowledgeModel


class KnowledgeModelPV(KnowledgeModel):

    def __init__(self, size_state: int, size_actions: int):
        self.size_state: int = size_state
        self.size_actions: int = size_actions
        self.model: keras.models.Model = self.build_model_learning()
        optimizer = keras.optimizers.Adam(learning_rate=0.2)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        # actions in one hot vector
        self.actions: np.array = np.eye(size_actions)

    def build_model_learning(self) -> keras.models.Model:
        state_input = keras.layers.Input(shape=(self.size_state,), name='state_input')
        state_zip = keras.layers.Dense(5, name='state_zip')(state_input)
        action_input = keras.layers.Input(shape=(self.size_actions,), name='action_input')
        action_zip = keras.layers.Dense(5, name='action_zip')(action_input)
        state_action = keras.layers.Concatenate(name='concatenate_state_action')([state_zip, action_zip])
        general_layer_output = keras.layers.Dense(1, name='q')(state_action)

        self.model = keras.models.Model(
            name='KnowledgeModelPV',
            inputs=[state_input, action_input],
            outputs=[general_layer_output],
        )
        return self.model

    def read_q_value(self,
                     s: np.array,  # state
                     a: int,  # action
                     ) -> float:
        _prediction = self.model.predict([
            np.expand_dims(s, axis=0),
            np.expand_dims(self.actions[a], axis=0)
        ])
        q = _prediction[0][0]
        return q

    def update_q_value(self,
                       s: np.array,  # state (normalmente seria el estado previo)
                       a: int,  # action
                       q: float,  # q_value
                       ):
        self.model.fit(
            x=[
                np.expand_dims(s, axis=0),
                np.expand_dims(self.actions[a], axis=0)
            ],
            y=np.array([q]),
            epochs=1,
            verbose=0,
        )

    def save_knowledge(self, filepath: str):
        keras.models.save_model(self.model, filepath)

    def load_knowledge(self, filepath: str):
        self.model = keras.models.load_model(filepath)


class QLearningAgentPV(QLearningAgent):

    def __init__(self,
                 env: Environment,
                 action_count: int,
                 ):
        self.knowledge_model: KnowledgeModelPV = KnowledgeModelPV(5, action_count)
        super().__init__(env, action_count, self.knowledge_model)

    def update_current_state(self) -> np.array:
        position: np.array = self.time_step.observation['position']
        velocity: np.array = self.time_step.observation['velocity']
        self.state_current = np.concatenate((position, velocity))
        return self.state_current
