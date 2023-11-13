import numpy as np
from dm_control import suite
from dm_control.rl.control import Environment
from agent import Agent

if __name__ == '__main__':
    random_state = np.random.RandomState(42)
    env: Environment = suite.load('cartpole', 'balance', task_kwargs={'random': random_state})

    ag: Agent = Agent(
        env=env,
        action_count=11,
        state_frames_count=4,
        frames_shape=(100, 100, 3),
    )
    ag.epsilon = .99
    ag.run_episode()
