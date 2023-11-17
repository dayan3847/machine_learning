from dm_control.rl.control import Environment

from dayan3847.reinforcement_learning.deep_mind.agent.Agent import Agent


class RandomAgent(Agent):
    def __init__(self,
                 env: Environment,
                 action_count: int,
                 ):
        super().__init__(env, action_count)
        (self.time_step, self.state_pre, self.state_current, self.step) = self.init_episode()

    def select_an_action(self) -> tuple[int, float, bool]:  # action, is_random
        return self.select_an_action_random()
