import gym, glob, os
from gym import error, spaces, utils
import pandas as pd
import numpy as np
import random

# actions
ACTION_SKIP = 0
ACTION_BET1 = 1
ACTION_BET2 = 2
#ACTION_BET3 = 3
#ACTION_BET4 = 4


class BetState:
	def __init__(self, data_path, sep=','):
		# load csv
		df = self.read_csv(data_path, sep=sep)
		self.df = df
		self.features = ['ratio_FGM', 'ratio_FGA', 'ratio_FGM3', 'ratio_FGA3', 'ratio_FTM', 'ratio_FTA', 'ratio_OR', \
						 'ratio_DR', 'ratio_Ast', 'ratio_TO', 'ratio_Stl', 'ratio_Blk', 'ratio_PF', 'diff_win',\
						 'diff_lose', 'ratio_cote']
		#self.features = ['Decimal1', 'Decimal']
		#self.features = ['ratio_FGM', 'ratio_FGA', 'ratio_FGM3', 'ratio_FGA3', 'ratio_FTM', 'ratio_FTA', 'ratio_OR', ]
		self.matches = np.arange(df.shape[0])
		random.shuffle(self.matches)
		self.index = 0

		
		print('imported data from {}'.format(data_path))
	    
	def read_csv(self, path, sep):
		df = pd.read_csv(path, sep=sep)
		return df
	
	def reset(self):
		# agent start at t=0
		idx = self.matches[self.index]
		data = self.df.iloc[idx] # contains all aggregated info of teams for that step
		values = data[self.features].values

		return values

	def next(self):
		# training stop there
		if self.index >= len(self.df) - 1:
			return self.index, None, True

		idx = self.matches[self.index]
		# return features for next time step
		data = self.df.iloc[idx]
		values = data[self.features].values

		# moving forward into time
		self.index += 1


		return self.index, values, False

	def shape(self):
		return self.df.shape

	def price(self):
		data = self.df.iloc[self.index-1]
		winner = ['winning_team']
		cote0  = ['Decimal1']
		cote1  = ['Decimal']
		winner = data[winner].values
		odds0 = data[cote0].values
		odds1 = data[cote1].values

		return winner, odds0, odds1

class BetEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	
	def __init__(self, datadir):
		self.bound = 100000
		self.money = 0
		self.amount1 = 5
		self.amount2 = 10
		self.inflation = 0
		self.dead = False
		self.state = None
		self.states = []
		
		for path in glob.glob(datadir + '/*.csv'):
			if not os.path.isfile(path):
				continue
		
			self.states.append(path)
		
		#self.observation_space = spaces.Box(low=-self.bound, high=self.bound, shape = (16,1))
		self.observation_space = spaces.Box(low=-self.bound, high=self.bound, shape=(16, 1))
		self.action_space = spaces.Discrete(3)
		
		if len(self.states) == 0:
			raise NameError('Invalid empty directory {}'.format(datadir))
	
	def step(self, action):
		assert self.action_space.contains(action)
		done = False
		prev_money = self.money
		
		price = self.state.price()

		if action == ACTION_BET1:
			if price[0] == 0 & np.random.choice(2,1, p=[0.9,0.1])==0:
				self.money += price[1]*self.amount2 - self.amount2

			else:
				self.money -= self.amount2

		elif action == ACTION_BET2:
			if price[0] == 1 & np.random.choice(2,1, p=[0.1,0.9])==1:
				self.money += price[2] * self.amount2 - self.amount2

			else:
				self.money -= self.amount2

		#elif action == ACTION_BET3:
		#	if price[0] == 0 & np.random.choice(2,1, p=[0.9,0.1])==0:
		#		self.money += price[1]*self.amount1 - self.amount1

		#	else:
		#		self.money -= self.amount1

		#elif action == ACTION_BET4:
		#	if price[0] == 1 & np.random.choice(2,1, p=[0.1,0.9])==1:
		#		self.money += price[2] * self.amount1 - self.amount1
		#	else:
		#		self.money -= self.amount1

		#if self.money <= self.amount2:
		#	done = True

		index, state, done_next = self.state.next()
		state = np.append(state, self.money)
		if not done:
			reward = self.money - prev_money
		else:
			reward = None
			if self.money < self.amount2:
				print('You died!')
			else:
				print('You reached the final line, but how much money do you have?')
			return index, state, price, reward, done, self.money



		
		return index, state, price, reward, done_next, self.money
	
	def reset(self):
		self.state = BetState(self.states[0])
		self.money = 3000
		self.dead = False
		
		index, state, done = self.state.next()
		state = np.append(state, self.money)
		return state
	

	
