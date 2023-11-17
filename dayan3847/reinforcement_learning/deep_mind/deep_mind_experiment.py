from datetime import datetime
import numpy as np
from dm_env import StepType

from dayan3847.reinforcement_learning.deep_mind.agent.Agent import Agent


def deep_mind_experiment(ag: Agent, model_name: str, knowledge_extension: str = 'csv'):
    while True:
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
            _r, _a, _is_random = ag.run_step()
            history_reward.append(_r)
            # history_frames.append(ag.env.physics.render(camera_id=0))
            print("Action: ", '\033[91m' if _is_random else '\033[92m', _a, '\033[00m')
            print("Reward: ", _r)
            history_position.append(ag.time_step.observation['position'])
            history_velocity.append(ag.time_step.observation['velocity'])

        print('saving knowledge')
        ag.save_knowledge(f'ep/{name}_knowledge.{knowledge_extension}')
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
