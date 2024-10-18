import sys,os
from calendar import day_abbr
from inference import representation
os.chdir('./../')
sys.path[0] = os.getcwd()
import pickle
import gym
import numpy as np
from IPython import display
from matplotlib import pyplot as plt

from utils import make_vec_envs, make_env


def show_state(env, step=0, changes=0, total_reward=0, name="", game="" , rep="" ):
    fig = plt.figure(10)
    plt.clf()
    plt.title("{} | Step: {} Changes: {} Total Reward: {}".format(name, step, changes, total_reward))
    plt.axis('off')
    plt.imshow(env.render(mode='rgb_array'))
    os.makedirs(f'./images/{game}/{rep}/', exist_ok=True)
    plt.savefig(f"./images/{game}/{rep}/{step}.jpg")
    plt.show()
    # display.clear_output(wait=True)
    display.display(plt.gcf())


if __name__ == '__main__':
    game = 'binary'
    dataset = 'wide'
    
    kwargs = {
    'change_percentage': 1,
    'verbose': False,
    'num_episodes': 3000
    }

    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10

    #change this base on the file you want to load
    env_name = '{}-{}-v0'.format(game, dataset)

    with open(f'./dataset/{env_name}.pkl', 'rb') as f:
        trajectories = pickle.load(f)

    env = make_vec_envs(env_name, dataset, None, 1, **kwargs)
    env.reset()
    for i in range(trajectories[0]['observations'].shape[0]):
        env.set_observation(trajectories[0]['observations'][i][0])
        show_state(env, i, changes=trajectories[0]['actions'][i], total_reward=trajectories[0]['rewards'][i], game=game, rep=dataset)
    env.set_observation(trajectories[0]['next_observations'][-1][0])
    show_state(env, game='sokoban', rep='wide',step = trajectories[0]['observations'].shape[0])

