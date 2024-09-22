"""
Run a trained agent and get generated maps
"""
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


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'next_observations': []
            }


def append_data(data, s, a, r, new_s, done):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['next_observations'].append(new_s)


def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


# def show_state(env, step=0, changes=0, total_reward=0, name=""):
#     plt.figure(10)
#     plt.clf()
#     plt.title("{} | Step: {} Changes: {} Total Reward: {}".format(name, step, changes, total_reward))
#     plt.axis('off')
#     # plt.imshow(env.render(mode='rgb_array'))
#     # plt.savefig(f"./runs/{step}.jpg")
#     # plt.show()
#     display.clear_output(wait=True)
#     display.display(plt.gcf())


def infer(game, representation, model_path, num_episode):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    policy = CustomPolicyBigMap

    agent = PPO2.load(model_path, custom_objects={
        "FullyConvPolicy": FullyConvPolicyBigMap,
        "FullyConvPolicySmallMap": FullyConvPolicySmallMap,
        "CustomPolicyBigMap": CustomPolicyBigMap,
        "CustomPolicySmallMap": CustomPolicySmallMap
    })
    # dir_path = r"./runs"
    # shutil.rmtree(dir_path, ignore_errors=True)
    # os.mkdir(dir_path)
    episode = 0
    data = []
    while episode < num_episode:
        print(episode)
        data_episodes = reset_data()
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
        # show_state(env)
        # obs_original = env.ori
        dones = False
        total_rewards = 0
        old_change = 0
        while not dones:
            action, _ = agent.predict(obs)
            new_obs, rewards, dones, info = env.step(action)
            h=5
            w=5
            dim=5
            _, _, action = np.unravel_index(action[0], (h, w, dim))
            # print(info)
            info = info[0]
            # show_state(env, info['iterations'], info['changes'], total_rewards)
            if (info["changes"] != old_change) or (dones and info["path-length"] > 18 and info["changes"] != old_change):
                print(info)
                append_data(data_episodes, obs, action, int(rewards), new_obs, dones)
                print(info)
                if dones and info["path-length"] > 18:
                    # print("finish")
                    break
            obs = new_obs
            total_rewards += rewards
            old_change = info["changes"]
            if dones:
                break
        if dones and info["path-length"] >= 18:
            data_episodes['observations'] = np.concatenate([np.array(data_episodes['observations'])], axis=0)
            data_episodes['next_observations'] = np.concatenate([np.array(data_episodes['next_observations'])],
                                                                axis=0)
            data_episodes['actions'] = np.concatenate([np.array(data_episodes['actions'])], axis=0)
            data_episodes['terminals'] = np.concatenate([np.array(data_episodes['terminals'])], axis=0)
            data_episodes['rewards'] = np.concatenate([np.array(data_episodes['rewards'])], axis=0)

            data.append(data_episodes)
            episode += 1
        if kwargs.get('verbose', False):
            print(info)
    fname = 'sokoban_wide_100000.pkl'
    with open(fname, 'wb') as f:
        print('Before dump')
        pickle.dump(data, f)


################################## MAIN ########################################
games = [] 
game = 'binary'
representation = 'wide'
# model_path = 'models/{}/{}/model_1.pkl'.format(game, representation)
model_path = './models/binary/wide/model_1.pkl'
num_episode = 1


if __name__ == '__main__':
    infer(game, representation, model_path, num_episode)