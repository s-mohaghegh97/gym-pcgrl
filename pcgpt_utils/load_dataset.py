import pickle

import gym
import numpy as np
from IPython import display
from matplotlib import pyplot as plt

from utils import make_vec_envs, make_env


def show_state(env, step=0, changes=0, total_reward=0, name=""):
    fig = plt.figure(10)
    plt.clf()
    plt.title("{} | Step: {} Changes: {} Total Reward: {}".format(name, step, changes, total_reward))
    plt.axis('off')
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    plt.savefig(f"/home/sajjad/Documents/PCG/gym-pcgrl/images/{step}.jpg")
    display.clear_output(wait=True)
    display.display(plt.gcf())

with open('/home/sajjad/Documents/PCG/gym-pcgrl/pcgpt_utils/sokoban_wide_100000.pkl', 'rb') as f:
    trajectories = pickle.load(f)

kwargs = {
    'change_percentage': 1,
    'verbose': False,
    'num_episodes': 3000
}
env_name = 'sokoban'
dataset = 'wide'
kwargs['cropped_size'] = 10
env_name = '{}-{}-v0'.format(env_name, dataset)
env = make_vec_envs(env_name, dataset, None, 1, **kwargs)
env.reset()
for i in range(trajectories[0]['observations'].shape[0]):
    env.set_observation(trajectories[0]['observations'][i][0])
    show_state(env, i, changes=trajectories[0]['actions'][i], total_reward=trajectories[0]['rewards'][i])
env.set_observation(trajectories[0]['ne_observations'][-1][0])
show_state(env)

