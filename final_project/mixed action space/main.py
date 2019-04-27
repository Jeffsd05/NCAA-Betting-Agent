import gym, random, torch
from config import load_config
from core2 import DDPG, OrnsteinUhlenbeckProcess, to_scalar
import numpy as np
from itertools import count
from tqdm import tqdm
import matplotlib.pyplot as plt
from gym_bet.envs import bet_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cf = load_config('config/config.py')
env = gym.make('Bet-v0')

cf.state_dim = env.observation_space.shape[0]

# tuple action space
cf.action_dim = env.action_space.spaces[0].n
cf.param_dim = 1

print('Trying environment Bet-v0')
print('State space size: ', cf.state_dim)
print('Action space size: ', env.action_space.spaces)
print('Parameter low: ', env.action_space.spaces[1].low)
print('Parameter high: ', env.action_space.spaces[1].high)

noise_process = OrnsteinUhlenbeckProcess(cf)
model = DDPG(cf)
model.copy_weights(model.actor, model.actor_target)
model.copy_weights(model.critic, model.critic_target)

losses = []
total_timesteps = 0
total_reward = []
for epi in tqdm(range(cf.max_episodes)):
	s_t = env.reset()
	s_t = torch.tensor(s_t.astype('float')).float().to(device)
	noise_process.reset()
	avg_reward = 0
	for t in tqdm(range(cf.max_step)):
		ad_t, p_t = model.sample_action(s_t)
		p_t = p_t.add(noise_process.sample()[0])
		if random.random() > cf.epsilon:
			a_t = torch.cat([ad_t, p_t])
			if p_t < 0.01:
				a = (int(ad_t.argmax()), np.array([0.01]))
			elif p_t  > 1:
				a = (int(ad_t.argmax()), np.array([1]))
			else:
				a = (int(ad_t.argmax()), p_t.detach().numpy())
		else:
			ad_t = torch.rand_like(ad_t)
			ad_t = ad_t/ad_t.sum()
			p_t = np.array([random.uniform(0.01,1)]).astype('float')
			a_t = torch.cat([ad_t, torch.tensor(p_t).float().to(device)]).to(device)
			a = (int(ad_t.argmax()), p_t)
			
		s_tp1, r_t, done, info = env.step(a)
		model.buffer.add(torch.tensor(s_t), a_t, torch.tensor(r_t), torch.tensor(s_tp1.astype(float)), torch.tensor(float(done == False)))
		avg_reward += r_t
	
		if done:
			break
		else:
			s_t = s_tp1
			s_t = torch.tensor(s_t.astype('float')).float().to(device)

		if model.buffer.len >= cf.replay_start_size:
			_loss_a, _loss_c = model.train_batch()
			losses.append([_loss_a, _loss_c])
			
	if len(losses) > 0:
		total_timesteps += t
		avg_loss_a, avg_loss_c = np.asarray(losses)[-100:].mean(0)
		print(
			'Episode {}: actor loss: {} critic loss: {}\
			total_reward: {} timesteps: {} tot_timesteps: {}'.format(
			epi, avg_loss_a, avg_loss_c, avg_reward, t, total_timesteps
			))
	print(a_t, a)
	cf.epsilon = cf.epsilon * cf.decay
	
	total_reward.append(avg_reward)
		
	if (epi + 1) % 100 == 0:
		model.save_models()
print('Completed training!')

plt.figure(1)	
plt.plot(total_reward)
plt.xlabel('Episodes')
plt.ylabel('Rewards')

plt.figure(2)
plt.plot(avg_loss_a)
plt.title('Actor Loss')
plt.xlabel('Episodes')
plt.ylabel('Loss')	
	
plt.figure(3)
plt.plot(avg_loss_c)
plt.title('Critic Loss')
plt.xlabel('Episodes')
plt.ylabel('Loss')	

plt.show()
