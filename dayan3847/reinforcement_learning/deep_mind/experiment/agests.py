import numpy as np

from dayan3847.models.Model import Model
from dayan3847.models.multivariate.MultivariateGaussianModel import MultivariateGaussianModel
from dayan3847.reinforcement_learning.agent.TemporalDifferenceLearningAgent import TemporalDifferenceLearningAgent
from dayan3847.reinforcement_learning.agent.TemporalDifferenceLearningAgentGaussian import QLearningAgentGaussian


def balance_example_5_11111() -> TemporalDifferenceLearningAgent:
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

    return QLearningAgentGaussian(
        action_count=action_count,
        models=models,
    )
