import gym
import gym_bet
import random

env = gym.make('Bet-v0')

obs = env.reset()
print(obs)

#for i in range(10):
	#obs = env.step(1)
	#print(obs)

done = False
while not done:
	action = random.choice([0,1,2])
	obs = env.step(action)
	done = obs[3]
	print(obs)
