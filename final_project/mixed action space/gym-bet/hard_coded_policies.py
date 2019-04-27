import gym
import gym_bet
import random
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm

env = gym.make('Bet-v0')

runs = 1000
final_step = []
for run in range(runs):
	obs = env.reset()
	done = False
	reward = []
	cash = []
	while not done:
		action = random.choice([0,1,2])
		obs = env.step(action)
		done = obs[3]
		reward.append(obs[2])
		cash.append(obs[4])
	final_step.append(len(cash)-1) 
avg_step = np.array(final_step).sum()/runs
print(final_step, avg_step)

plt.figure(1)
plt.subplot(211)
plt.title('Random Policy')
plt.plot(reward)
plt.ylabel('Reward')
plt.subplot(212)
plt.plot(cash)
plt.xlabel('Steps')
plt.ylabel('Money')

plt.figure(2)
plt.plot(final_step)
plt.xlabel('run')
plt.ylabel('step of death')
plt.title('Step of death for random policy')

#  odds policy
obs = env.reset()
done = False
reward_odds_policy = []
cash_odds_policy = []
action = 1
while not done:
	obs = env.step(action)
	done = obs[3]
	action = np.argmin(obs[5][0:2])
	reward_odds_policy.append(obs[2])
	cash_odds_policy.append(obs[4])
	
plt.figure(3)
plt.subplot(211)
plt.title('Odds Policy')
plt.plot(reward_odds_policy)
plt.ylabel('Reward')
plt.subplot(212)
plt.plot(cash_odds_policy)
plt.xlabel('Steps')
plt.ylabel('Money')
plt.show()
