import gym
import gym_bet
import random
import matplotlib.pyplot as plt 
import numpy as np

env = gym.make('Bet-v0')

obs = env.reset()
print(obs)

action = (1, np.array([1000]))
for i in range(10):
	obs = env.step(action)
	print(obs)





