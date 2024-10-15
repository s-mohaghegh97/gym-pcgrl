import os
import pickle
import random
import shutil

import numpy as np

import model
from stable_baselines import PPO2

import time

from utils import make_vec_envs
from model import FullyConvPolicyBigMap, FullyConvPolicySmallMap, CustomPolicyBigMap, CustomPolicySmallMap

from IPython import display
import matplotlib.pyplot as plt

from stable_baselines.common.policies import FeedForwardPolicy

class FullyConvPolicyBigMap(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(FullyConvPolicyBigMap, self).__init__(*args, **kwargs,
                                                   net_arch=[dict(pi=[64, 64], vf=[64, 64])])

class Inference:
    def __init__(self, game, representation):
        # Set dimensions based on game and representation
        if game == "binary":
            self.dim = 2
            self.width = 14
            self.height = 14
            self.length = 'path-length'
        elif game == "zelda":
            self.dim = 8
            self.width = 11
            self.height = 7
            self.length = 'path-length'

        elif game == "sokoban":
            self.dim = 5
            self.width = 5
            self.height = 5
            self.length = 'sol-length'
        else:
            raise ValueError("Unsupported game type")


    def reset_data(self):
        return {'observations': [],
                'actions': [],
                'terminals': [],
                'rewards': [],
                'next_observations': []
                }

    def append_data(self, data, s, a, r, new_s, done):
        data['observations'].append(s)
        data['actions'].append(a)
        data['rewards'].append(r)
        data['terminals'].append(done)
        data['next_observations'].append(new_s)

    def infer(self, game, representation, model_path, num_episode):
        # policy = CustomPolicyBigMap
        agent = PPO2.load(model_path)
        episode = 0
        data = []
        while episode < num_episode:
            print(episode)
            data_episodes = self.reset_data()
            kwargs = {
                'change_percentage': random.random(),
                'verbose': False,
            }
            env_name = '{}-{}-v0'.format(game, representation)
            if game == "binary":
                model.FullyConvPolicy = model.FullyConvPolicyBigMap
                kwargs['cropped_size'] = 28
            elif game == "zelda":
                model.FullyConvPolicy = model.FullyConvPolicyBigMap
                kwargs['cropped_size'] = 22
            elif game == "sokoban":
                model.FullyConvPolicy = model.FullyConvPolicySmallMap
                kwargs['cropped_size'] = 10
            kwargs['render'] = False
            env = make_vec_envs(env_name, representation, None, 1, **kwargs)
            obs = env.reset()
            dones = False
            total_rewards = 0
            old_change = 0
            while not dones:
                action, _ = agent.predict(obs)
                new_obs, rewards, dones, info = env.step(action)
                print(action)
                _, _, action = np.unravel_index(action[0], (self.height, self.width, self.dim))
                info = info[0]
                if (info["changes"] != old_change) or (dones and info[ self.length] > 18 and info["changes"] != old_change):
                    self.append_data(data_episodes, obs, action, int(rewards), new_obs, dones)
                    if dones and info[ self.length] > 18:
                        break
                obs = new_obs
                total_rewards += rewards
                old_change = info["changes"]
                if dones:
                    break
            if dones and info[ self.length] >= 18:
                data_episodes['observations'] = np.concatenate([np.array(data_episodes['observations'])], axis=0)
                data_episodes['next_observations'] = np.concatenate([np.array(data_episodes['next_observations'])], axis=0)
                data_episodes['actions'] = np.concatenate([np.array(data_episodes['actions'])], axis=0)
                data_episodes['terminals'] = np.concatenate([np.array(data_episodes['terminals'])], axis=0)
                data_episodes['rewards'] = np.concatenate([np.array(data_episodes['rewards'])], axis=0)

                data.append(data_episodes)
                episode += 1
        fname = f'./dataset/{game}_{representation}_dataset.pkl'
        if not os.path.exists('./dataset'):  
            os.mkdir('./dataset')
        with open(fname, 'wb') as f:
            print('Before dump')
            pickle.dump(data, f)

################################## MAIN ########################################
if __name__ == '__main__':
    game = 'zelda'
    representation = 'wide'
    model_path = 'runs/{}_{}_1_log/latest_model.pkl'.format(game, representation)
    num_episode = 1

    inference = Inference(game, representation)
    inference.infer(game, representation, model_path, num_episode)
