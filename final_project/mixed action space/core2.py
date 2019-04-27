import torch, random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_scalar(arr):
    return [x.cpu().data.tolist()[0] for x in arr]

def to_tensor(arr, batch_size):
	try:
		dim = len(arr[0])
	except TypeError:
		dim = 1
		
	T = torch.zeros(batch_size, dim)
	for i in range(batch_size):
		T[i] = arr[i]
	return T.to(device)


class ReplayBuffer:
	def __init__(self, cf):
		self.buffer_size = cf.capacity
		self.len = 0
		self.batch_size = cf.batch_size
		
		# Create buffers for (s_t, a_t, r_t, s_t+1, term)
		self.buffer = deque(maxlen=self.buffer_size)
		
	def sample(self, count):
		count = min(count, self.len)
		batch = random.sample(self.buffer, count)
		s_t, a_t, r_t, s_tp1, term = zip(*batch)
		s_t = to_tensor(s_t, count)
		a_t = to_tensor(a_t, count)
		r_t = to_tensor(r_t, count)
		s_tp1 = to_tensor(s_tp1, count)
		term = to_tensor(term, count)
		
		return s_t, a_t, r_t, s_tp1, term
	
	def add(self, s_t, a_t, r_t, s_tp1, term):
		transition = (s_t, a_t, r_t, s_tp1, term)
		self.len += 1
		if self.len > self.buffer_size:
			self.len = self.buffer_size
		self.buffer.append(transition)
		
class OrnsteinUhlenbeckProcess():
	def __init__(self, cf):
		self.param_dim = cf.param_dim

		self.mu_val = cf.mu
		self.sigma_val = cf.sigma
		self.theta = cf.theta
		self.dt = cf.dt

	def reset(self):
		self.mu = np.zeros(self.param_dim, dtype='float') + self.mu_val
		self.sigma = np.ones(self.param_dim, dtype='float') * self.sigma_val
		self.X = np.zeros_like(self.mu)

	def sample(self):
		epsilon = np.random.normal(size=self.mu.shape).astype('float')
		term1 = self.theta * (self.mu - self.X) * self.dt
		term2 = self.sigma * np.sqrt(self.dt) * epsilon
		self.X += term1 + term2
		return self.X

class Actor(nn.Module):

	def __init__(self, cf):
		super(Actor, self).__init__()
		self.act = nn.ReLU()
		self.lin1 = nn.Linear(cf.state_dim, 200)
		self.bn1 = nn.BatchNorm1d(200)
		self.lin2 = nn.Linear(200, 100)
		self.bn2 = nn.BatchNorm1d(100)
		self.lin_d = nn.Linear(100, cf.action_dim)
		self.lin_c = nn.Linear(100, cf.param_dim)
		self.class_prob = nn.Softmax()
		self.bounded_param = nn.Sigmoid()
		
	def forward(self, state):
		x = self.act(self.lin1(state))
		x = self.act(self.lin2(x))
		x1 = self.class_prob(self.lin_d(x))
		x2 = self.bounded_param(self.lin_c(x))

		return x1, x2
	
class Critic(nn.Module):

	def __init__(self, cf):
		super(Critic, self).__init__()
		self.transform_state = nn.Sequential(
			nn.Linear(cf.state_dim, 200),
			nn.BatchNorm1d(200),
			nn.ReLU()
			)
		nn.init.xavier_uniform_(self.transform_state[0].weight.data)

		self.transform_both = nn.Sequential(
			nn.Linear(200 + cf.action_dim + cf.param_dim, 300),
			nn.BatchNorm1d(300),
			nn.ReLU(),
			nn.Linear(300, 1)
			)
		nn.init.xavier_uniform_(self.transform_both[0].weight.data)
		self.transform_both[-1].weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state, action):
		state = self.transform_state(state)
		both = torch.cat([state, action], 1)
		
		return self.transform_both(both)

class DDPG(nn.Module):
	def __init__(self, cf):
		super(DDPG, self).__init__()
		self.cf = cf

		self.actor = Actor(cf).to(device)
		self.actor_target = Actor(cf).to(device)
		self.actor_optimizer = optim.Adam(
			self.actor.parameters(), lr=cf.actor_learning_rate
		)

		self.critic = Critic(cf).to(device)
		self.critic_target = Critic(cf).to(device)
		self.critic_optimizer = optim.Adam(
			self.critic.parameters(), lr=cf.critic_learning_rate
		)

		self.buffer = ReplayBuffer(cf)

	def update_targets(self, model, target):
		''''
		Soft updates.
		'''
		for p, target_p in zip(model.parameters(), target.parameters()):
			target_p.data.copy_(
				self.cf.tau * p.data + (1-self.cf.tau) * target_p.data
			)

	def copy_weights(self, model, target):
		for p, target_p in zip(model.parameters(), target.parameters()):
			target_p.data.copy_(
				p.data
			)

	def sample_action(self, state):
		return self.actor(state)
		
	def train_batch(self):
		s_t, a_t, r_t, s_tp1, term = self.buffer.sample(self.cf.batch_size)
		# The below 2 operations need to be detached since we only
		# update critic and not targets
		ad_tp1, p_tp1 = self.actor_target(s_tp1)
		a_tp1 = torch.cat([ad_tp1, p_tp1], 1)
		q_value = self.critic_target(s_tp1, a_tp1).squeeze()
		td_target = r_t + self.cf.gamma * term * q_value
		td_target.requires_grad_()
		td_current = self.critic(s_t, a_t).squeeze()

		#critic_loss = F.smooth_l1_loss(td_current, td_target)
		critic_loss = F.mse_loss(td_current, td_target)

		self.critic_optimizer.zero_grad()
		critic_loss.backward(retain_graph=True)
		self.critic_optimizer.step()

		ad_t_pred, p_t_pred = self.actor(s_t)
		a_t_pred = torch.cat([ad_t_pred, p_t_pred], 1)
		q_pred = self.critic(s_t, a_t_pred)
		actor_loss = -1 * q_pred.mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.update_targets(self.actor, self.actor_target)
		self.update_targets(self.critic, self.critic_target)
		return actor_loss, critic_loss

	def save_models(self):
		torch.save(self.actor.state_dict(), 'models/best_actor.model')
		torch.save(self.critic.state_dict(), 'models/best_critic.model')
		torch.save(self.actor_target.state_dict(), 'models/best_actor_target.model')
		torch.save(self.critic_target.state_dict(), 'models/best_critic_target.model')


	def load_models(self):
		self.actor.load_state_dict(
			torch.load('models/best_actor.model'))
		self.critic.load_state_dict(
			torch.load('models/best_critic.model'))
