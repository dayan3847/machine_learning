import numpy as np
from dm_env import StepType, TimeStep

ACTIONS_COUNT: int = 4
BOARD_SHAPE: tuple[int, int] = (4, 12)


def fix_position(position: np.array, limits: tuple[int, int]) -> np.array:
    _count = len(position)
    if _count != len(limits):
        raise Exception('Invalid position or limits')
    for i in range(_count):
        if position[i] < 0:
            position[i] = 0
        elif position[i] >= limits[i]:
            position[i] = limits[i] - 1
    return position


class CliffWalkingEnvironment:
    def __init__(self):
        # BOARD
        # Add board
        self.board_reward: np.array = np.full(BOARD_SHAPE, -1)
        # Add cliff
        self.board_reward[-1, 1:-1] = -100
        # Add goal
        self.board_reward[-1, -1] = 100

        self.board_incidence: np.array = np.zeros(BOARD_SHAPE)
        # ACTIONS
        self.action_values: list[np.array] = [
            np.array([-1, 0]),  # up
            np.array([1, 0]),  # down
            np.array([0, -1]),  # left
            np.array([0, 1]),  # right
        ]
        # STATS
        self.count_win: int = 0
        self.count_lose: int = 0

        self.time_step: TimeStep = TimeStep(
            step_type=StepType.FIRST,
            observation={
                'position': np.array([3, 0]),
            },
            reward=None,
            discount=None,
        )

    def init_time_step(self):
        self.time_step = TimeStep(
            step_type=StepType.FIRST,
            observation={
                'position': np.array([3, 0]),
            },
            reward=None,
            discount=None,
        )

    def apply_action(self, action: int):
        # Validate action
        if action not in range(ACTIONS_COUNT):
            raise Exception('Invalid action')
        # Validate time_step
        if self.time_step.last():
            raise Exception('Invalid time_step')
        # Apply action
        position = self.time_step.observation['position'] + self.action_values[action]
        # Fix position
        position = fix_position(position, BOARD_SHAPE)
        # state as tuple
        state_tuple: tuple = tuple(position)
        # Get reward
        reward: float = self.board_reward[state_tuple]
        self.board_incidence[state_tuple] += 1
        episode_end: bool = False
        if abs(reward) == 100:
            episode_end = True
            if reward > 0:
                self.count_win += 1
            else:
                self.count_lose += 1

        self.time_step = TimeStep(
            step_type=StepType.LAST if episode_end else StepType.MID,
            observation={
                'position': position,
            },
            reward=reward,
            discount=None,
        )

    def run_episode(self, policy: callable) -> tuple[float, bool]:  # (reward_accumulated, win)
        r: float = 0
        self.init_time_step()
        while not self.time_step.last():
            action: int = policy(self.time_step)
            self.apply_action(action)
            r += self.time_step.reward
        policy(self.time_step)
        return r, self.time_step.reward == 100

    def run(self, policy: callable, episodes: int = 1):
        for e in range(episodes):
            print('episode: {}'.format(e))
            self.run_episode(policy)
