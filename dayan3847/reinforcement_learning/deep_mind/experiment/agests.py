import numpy as np

from dm_env import TimeStep

from dayan3847.models.Model import Model
from dayan3847.models.multivariate.MultivariateGaussianModel import MultivariateGaussianModel
from dayan3847.reinforcement_learning.agent.TemporalDifferenceLearningAgent import TemporalDifferenceLearningAgent
from dayan3847.reinforcement_learning.agent.TemporalDifferenceLearningAgentGaussian import QLearningAgentGaussian
from dayan3847.reinforcement_learning.agent.TemporalDifferenceLearningAgentTable import QLearningAgentTable


def get_state(time_step: TimeStep) -> np.array:
    position: np.array = time_step.observation['position']
    velocity: np.array = time_step.observation['velocity']
    state = np.concatenate((position, velocity))
    return state


LIMITS = [
    np.linspace(-.5, .63, 10),
    np.linspace(.2, 1, 10),
    np.linspace(-1, 1, 20),
    np.linspace(-1, 1, 20),
    np.linspace(-5, 5, 30),
]


def get_state_pos(time_step: TimeStep) -> np.array:
    state = get_state(time_step)
    for i in range(len(state)):
        state[i] = np.digitize(state[i], LIMITS[i])
        if state[i] != 0:
            state[i] -= 1

    return state


def balance_qlearning_gaussian_5_11111() -> tuple[TemporalDifferenceLearningAgent, callable]:
    action_count = 5
    models: list[Model] = [
        MultivariateGaussianModel(
            [
                (-2, 2, 1),
                (-1, 1, 1),
                (-1, 1, 1),
                (-2, 2, 1),
                (-15, 15, 1),
            ],
            cov=np.array([
                [2, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 2, 0],
                [0, 0, 0, 0, 5],
            ]),
            init_weights=0,
        ) for _ in range(action_count)
    ]

    return (
        QLearningAgentGaussian(
            action_count=action_count,
            models=models,
        ),
        get_state
    )


def balance_qlearning_table_5() -> tuple[TemporalDifferenceLearningAgent, callable]:
    action_count = 5
    ag = QLearningAgentTable(
        action_count=action_count,
        state_shape=(10, 10, 20, 20, 30),
    )
    ag.epsilon = .2
    ag.knowledge_model.load_knowledge()
    # ag.knowledge_model.reset_knowledge()
    return ag, get_state_pos


def balance_qlearning_table_6() -> tuple[TemporalDifferenceLearningAgent, callable]:
    action_count = 6
    ag = QLearningAgentTable(
        action_count=action_count,
        state_shape=(10, 10, 20, 20, 30),
    )
    ag.epsilon = .1
    ag.knowledge_model.load_knowledge()
    # ag.knowledge_model.reset_knowledge()
    return ag, get_state_pos
