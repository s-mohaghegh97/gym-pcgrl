import os
import pickle
import random
import shutil
import sys
os.chdir('./../')
sys.path[0] = os.getcwd()
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
            self._target_path = 20
            self.binary_first_path = 0
        
        elif game == "zelda":
            self.dim = 8
            self.width = 11
            self.height = 7
            self.length = 'path-length'
            self._target_enemy_dist = 4
            self._target_path = 16

        elif game == "sokoban":
            self.dim = 5
            self.width = 5
            self.height = 5
            self.length = 'sol-length'
            self._target_solution = 18

        else:
            raise ValueError("Unsupported game type")
    
    def check_done(self, game , info):
        if game == "sokoban":
            return info[self.length] >= self._target_solution
        elif game == "zelda":
            return info[self.length] > self._target_path and info["nearest-enemy"] >= self._target_enemy_dist
        elif game == "binary":
            print(info[self.length] , self.binary_first_path,info["regions"],self._target_path)
            return info["regions"] == 1 and info[self.length] - self.binary_first_path >= self._target_path

    def reset_data(self):
        return {'observations': [],
                'actions': [],
                'terminals': [],
                'rewards': [],
                'next_observations': [],
                'x': [],
                'y': []
                }

    def append_data(self, data, s, a, r, new_s, done, x, y):
        data['observations'].append(s)
        data['actions'].append(a)
        data['rewards'].append(r)
        data['terminals'].append(done)
        data['next_observations'].append(new_s)
        data['x'].append(x)
        data['y'].append(y)

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
            first_step = True
            info = None
            while not dones:
                action, _ = agent.predict(obs)
                new_obs, rewards, dones, info = env.step(action)
                print(action)
                _, _, action = np.unravel_index(action[0], (self.height, self.width, self.dim))
                info = info[0]
                if first_step: 
                    self.binary_first_path = info[self.length]
                    first_step = False
                #done base on the games like zelda
                # print(f'info: {info}')
                if (info["changes"] != old_change) or (self.check_done(game, info) and info["changes"] != old_change):
                    self.append_data(data_episodes, obs, action, int(rewards), new_obs, dones, info['x'], info['y'])
                    if dones and self.check_done(game, info):
                        break
                obs = new_obs
                total_rewards += rewards
                old_change = info["changes"]
                if dones:
                    break
            #done base on the games like sokoban
            if dones and self.check_done(game, info):
                data_episodes['observations'] = np.concatenate([np.array(data_episodes['observations'])], axis=0)
                data_episodes['next_observations'] = np.concatenate([np.array(data_episodes['next_observations'])], axis=0)
                data_episodes['actions'] = np.concatenate([np.array(data_episodes['actions'])], axis=0)
                data_episodes['terminals'] = np.concatenate([np.array(data_episodes['terminals'])], axis=0)
                data_episodes['rewards'] = np.concatenate([np.array(data_episodes['rewards'])], axis=0)
                data_episodes['x'] = np.concatenate([np.array(data_episodes['x'])], axis=0)
                data_episodes['y'] = np.concatenate([np.array(data_episodes['y'])], axis=0)
                data.append(data_episodes)
                episode += 1
        fname = f'./dataset/{game}-{representation}-v0.pkl'
        if not os.path.exists('./dataset'):  
            os.mkdir('./dataset')
        with open(fname, 'wb') as f:
            print('Before dump')
            pickle.dump(data, f)

################################## MAIN ########################################
if __name__ == '__main__':
    game = 'binary'
    representation = 'wide'
    model_path = './runs/{}_{}_1_log/best_model.pkl'.format(game, representation)
    num_episode = 1

    inference = Inference(game, representation)
    inference.infer(game, representation, model_path, num_episode)
