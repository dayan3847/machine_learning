import time
from datetime import datetime
import numpy as np
from dm_env import StepType
from dm_control.rl.control import Environment

from dayan3847.reinforcement_learning.deep_mind.agent.Agent import Agent


# TODO move to utils
def get_action_values(env_: Environment, action_count_: int) -> np.array:
    _spec = env_.action_spec()
    # return np.linspace(_spec.minimum, _spec.maximum, action_count_)
    if action_count_ != 7:
        raise Exception('for this momento only suport 7 actions')
    return np.array([
        -.6, -.3, -.1, 0, .1, .3, .6,
    ], dtype=_spec.dtype)


def deep_mind_experiment(ag: Agent, model_name: str, knowledge_extension: str = 'csv'):
    while True:
        start_time = time.time()
        if np.loadtxt('stop.txt') != 0:
            print('stop')
            break
        name: str = datetime.now().strftime('%Y%m%d%H%M%S')
        print(f'running episode {name}')
        ag.init_episode()
        history_reward: list[float] = []
        history_position: list[np.array] = []
        history_velocity: list[np.array] = []

        # history_frames: list[np.array] = [ag.env.physics.render(camera_id=0)]
        while StepType.LAST != ag.time_step.step_type:
            if np.loadtxt('stop.txt') == 2:
                print('\033[96m' + 'stop' + '\033[00m')
                break
            print('\033[96m{}\033[00m'.format(ag.step))
            _r, _a, _q, _is_random = ag.run_step()

            #         if time_step_prev is None:
            #     if self.time_step_prev
            #
            # if self.time_step.last():
            #     return None
            # self.step += 1
            # a, q, is_random = self.select_an_action()  # action
            # self.state_pre = self.state_current
            # self.time_step = self.env.step(float(self.action_values[a]))
            # self.state_current = self.update_current_state()
            # r: float = float(self.time_step.reward)
            # return r, a, q, is_random

            history_reward.append(_r)
            # history_frames.append(ag.env.physics.render(camera_id=0))
            history_position.append(ag.time_step.observation['position'])
            history_velocity.append(ag.time_step.observation['velocity'])

        print('saving knowledge')
        ag.save_knowledge(f'{model_name}_knowledge.{knowledge_extension}')
        ag.save_knowledge(f'ep/{name}_{model_name}_knowledge.{knowledge_extension}')
        print('saving reward')
        np.savetxt('reward.txt', history_reward)
        np.savetxt(f'ep/{name}_{model_name}_reward.txt', history_reward)
        print('saving position')
        np.savetxt('position.txt', history_position)
        np.savetxt(f'ep/{name}_{model_name}_position.txt', history_position)
        print('saving velocity')
        np.savetxt('velocity.txt', history_velocity)
        np.savetxt(f'ep/{name}_{model_name}_velocity.txt', history_velocity)
        # print('saving frames')
        # np.save('frames.npy', history_frames)

        # print in yellow
        print('\033[93m' + 'episode time: ' + str(time.time() - start_time) + '\033[00m')

    np.savetxt('stop.txt', [0])
