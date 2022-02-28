'''
Animation for the OpenAI Gym problem.
'''

import os
import math
import pickle
import gym
import numpy as np


def isDiscreteActions(env):
    return 'Discrete' in str(type(env.action_space))


# --- OpenAI Gym settings. --- #
# Only supports environments with the following state and action spaces:
# env.observation_space - Box(X,)
# env.action_space      - Box(X,) or Discrete(X)
envName = 'BipedalWalker-v3'

# Episode length should be longer than the termination condition defined in the gym environment.
episode_length = 100000

# Load MBEANN individual data
path = os.path.join(os.path.dirname(__file__), 'results_openai_gym_2147483648')
gen = '499'

with open('{}/data_ind_gen{:0>4}.pkl'.format(path, gen), 'rb') as pkl:
    ind = pickle.load(pkl)

# Make gym envirionment.
env = gym.make(envName)

total_reward = 0
observation = env.reset()

for t in range(episode_length):

    env.render()

    action = ind.calculateNetwork(observation)

    if isDiscreteActions(env):
        action = np.argmax(action)
    else:
        action = action * (env.action_space.high - env.action_space.low) + env.action_space.low

    observation, reward, done, info = env.step(action)

    total_reward += reward

    if done:
        print("Episode finished after {} timesteps with getting reward {}".format(t + 1, total_reward))
        break

env.close()
