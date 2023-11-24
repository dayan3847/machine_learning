import numpy as np
from dm_control.rl.control import Environment
from dm_env import StepType, TimeStep


def get_action_values(env_: Environment, action_count_: int) -> np.array:
    _spec = env_.action_spec()
    # return np.linspace(_spec.minimum, _spec.maximum, action_count_)
    if action_count_ != 7:
        raise Exception('for this momento only suport 7 actions')
    return np.array([
        -.6, -.3, -.1, 0, .1, .3, .6,
    ], dtype=_spec.dtype)


class Agent:
    def __init__(self,
                 env: Environment,
                 action_count: int,
                 ):
        self.env: Environment = env
        # Action:
        self.action_count: int = action_count
        self.action_values: np.array = get_action_values(env, action_count)

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

    def select_an_action(self) -> tuple[int, float, bool]:  # action, is_random
        pass

    def select_an_action_random(self) -> tuple[int, float, bool]:  # action, q_value, is_random
        return np.random.randint(self.action_count), 0, True

    def run_step(self) -> tuple[float, int, float, bool] | None:
        if StepType.LAST == self.time_step.step_type:
            return None
        self.step += 1
        a, q, is_random = self.select_an_action()  # action
        self.state_pre = self.state_current
        self.time_step = self.env.step(float(self.action_values[a]))
        self.state_current = self.update_current_state()
        r: float = float(self.time_step.reward)
        return r, a, q, is_random

    def save_knowledge(self, filepath: str):
        pass
