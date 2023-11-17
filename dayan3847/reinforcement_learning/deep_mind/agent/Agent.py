import numpy as np
from dm_control.rl.control import Environment
from dm_env import StepType, TimeStep


class Agent:
    def __init__(self,
                 env: Environment,
                 action_count: int,
                 ):
        self.env: Environment = env
        spec = env.action_spec()
        # Action:
        self.action_count: int = action_count
        self.action_values: np.array = np.linspace(spec.minimum, spec.maximum, action_count)

    def init_episode(self):
        self.time_step: TimeStep = self.env.reset()
        self.state_current: np.array = self.update_current_state()
        self.state_pre: np.array = self.state_current
        self.step: int = 0
        return self.time_step, self.state_pre, self.state_current, self.step

    def update_current_state(self) -> np.array:
        position: np.array = self.time_step.observation['position']
        velocity: np.array = self.time_step.observation['velocity']
        self.state_current = np.concatenate((position, velocity))
        return self.state_current

    def select_an_action(self) -> tuple[int, bool]:  # action, is_random
        pass

    def select_an_action_random(self) -> int:
        return np.random.randint(self.action_count)

    def run_step(self) -> tuple[float, int, bool] | None:
        if StepType.LAST == self.time_step.step_type:
            return None
        self.step += 1
        a, is_random = self.select_an_action()  # action
        self.state_pre = self.state_current
        self.time_step = self.env.step(float(self.action_values[a]))
        self.state_current = self.update_current_state()
        r: float = float(self.time_step.reward)
        return r, a, is_random

    def save_knowledge(self, filepath: str):
        pass
