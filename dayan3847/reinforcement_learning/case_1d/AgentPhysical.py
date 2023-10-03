import numpy as np
from abc import ABC

from dayan3847.reinforcement_learning.case_1d import Agent


class AgentPhysical(Agent, ABC):
    # Position of the agent
    point: np.array
    color: tuple[int, int, int]
