import gym
from config import load_config
from modules import DDPG, to_scalar
import numpy as np
import imageio


cf = load_config('config/config.py')
env = gym.make('Bet-v0')

cf.stat_dim = env.observation_space.shape[0]
cf.action_dim = env.action_space.spaces[0].n + 1 # because action_space is tuple

print('Trying environment Bet-v0')
print('State Dimensions: ', cf.state_dim)
print('Action Dimensions: ', cf.action_dim)
print('Action low: ', env.action_space.spaces[1].low)
print('Action high: ', env.action_space.spaces[1].high)

model = DDPG(cf)
model.load_models()

for epi in range(1):
    s_t = env.reset()
    avg_reward = 0
    while True:
        a_t = model.sample_action(s_t)
        s_t, r_t, done, info = env.step(a_t)
        avg_reward += r_t

        if done:
            break

print('Completed testing!')
