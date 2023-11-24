from dayan3847.reinforcement_learning.deep_mind.agent.Agent import Agent


class RandomAgent(Agent):

    def select_an_action(self,
                         s=None,
                         a=None,
                         ) -> tuple[int, float, bool]:  # action, is_random
        return (self.select_an_action_random(), 0, True) if a is None \
            else (a, 0, False)
