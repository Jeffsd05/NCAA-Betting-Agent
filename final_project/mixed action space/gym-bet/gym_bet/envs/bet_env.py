import gym, glob, os
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import numpy as np

# discrete action
skip = 0
bet1 = 1
bet2 = 2

class BetState:
	def __init__(self, data_path, sep=','):
		
		# load csv
		df = self.read_csv(data_path, sep=sep)
		self.df = df.sample(frac=1)
		self.features = ['ratio_FGM', 'ratio_FGA', 'ratio_FGM3', 'ratio_FGA3', 'ratio_FTM', 'ratio_FTA', 'ratio_OR', 'ratio_DR', 'ratio_Ast', 'ratio_TO', 'ratio_Stl', 'ratio_Blk', 'ratio_PF', 'odds1', 'odds2']
		self.game_param = ['odds1','odds2','winner']
		
		# keep track of time
		self.index = 0
		
		print('imported data from {}'.format(data_path))
	    
	def read_csv(self, path, sep):
		df = pd.read_csv(path, sep=sep)
		return df
	
	def reset(self):
		values = self.df.loc[0, self.features].values

	def next(self):
		
		# training stop there
		if self.index >= len(self.df) - 1:
			return None, True
		
		# return features for next time step
		values = self.df.loc[self.index, self.features].values
		
		# moving forward into time
		self.index += 1
		
		return values, False

	def price(self):
		return self.df.loc[self.index - 1, self.game_param].values
	
class BetEnv(gym.Env):
	metadata = {'render.modes': ['human']}
	
	def __init__(self, datadir):
		self.money = 0
		self.inflation = 0.1
		self.dead = False
		self.state = None
		self.states = []
		
		for path in glob.glob(datadir + '/*.csv'):
			if not os.path.isfile(path):
				continue
		
			self.states.append(path)
		
		self.observation_space = spaces.Box(0, 1e400, shape=(15,), dtype=np.float32)
		self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Box(0.01, 1, shape=(1,), dtype=np.float32)))
		
		if len(self.states) == 0:
			raise NameError('Invalid empty directory {}'.format(dirname))
	
	def step(self, action):
		assert self.action_space.contains(action) # action input should be a tuple
		
		prev_money = self.money
		
		price = self.state.price()
		
		if self.money <= 0:
			self.dead = True
		
		if action[0] == skip:
			self.money -= self.inflation
		elif action[0] == bet1:
			if price[2] == 0:
				self.money += price[0] * action[1][0] - action[1][0]
			else:
				self.money -= action[1][0]
		elif action[0] == bet2:
			if price[2] == 1:
				self.money += price[1] * action[1][0] - action[1][0]
			else:
				self.money -= action[1][0]

		state, done = self.state.next()
		if self.dead == True:
			done = True
		
		if not done:
			reward = self.money - prev_money
		else:
			reward = 0
			if self.money < 0:
				print('You died!')
			else:
				print('You reached the final line, but how much money do you have?')
				
		
		return state, reward, done, self.money
	
	def reset(self):
		self.state = BetState(self.states[0])
		self.money = 1000
		self.dead = False
		state = self.state.next()
		return state[0]
	
	def render(self, mode='human',close=False):
		pass
	
