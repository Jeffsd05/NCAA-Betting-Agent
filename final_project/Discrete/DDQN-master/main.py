from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import deque

import agent
import gym
import observer
from parameters import *
import bet_env
import numpy as np
import matplotlib.pyplot as plt
EPSILON_MIN = 0.05
EPSILON_MAX = 1.0
EPSILON_DECAY = 0.00001
MEMORY_CAPACITY = 500000
TARGET_UPDATE = 100
SIZE_HIDDEN = 16
LEARNING_RATE = 15
GAMMA = 0.99
LEARNING_RATE = 0.0075
MAX_STEPS = 2000
ACTIVATION = 'relu'
LEARNING_START = 100
N_EPISODES = 500

class Experiment:
    def __init__(self, path):
        self.env = bet_env.BetEnv(path)
        self.episode_count = 0
        self.reward_buffer = deque([], maxlen=100)
        self.wallet = []
        self.rewards = []
        self.steps = []

    def run_experiment(self, agent):
        for n in range(N_EPISODES):
            self.run_episode(agent)
        plt.plot(self.steps)
        plt.xlabel('Number of episode')
        plt.ylabel('Number of step before bankruptcy')
        plt.title('DDQN learning curve')
        plt.savefig('DDQN learning curve - step' + '.png')
        plt.clf()

        plt.plot(self.rewards)
        plt.xlabel('Number of episode')
        plt.ylabel('Rewards after episode')
        plt.title('DDQN learning curve')
        plt.savefig('DDQN learning curve - reward' + '.png')
        plt.clf()

        plt.plot(self.wallet)
        plt.xlabel('Number of episode')
        plt.ylabel('wallet after episode')
        plt.title('DDQN learning curve')
        plt.savefig('DDQN learning curve - wallet' + '.png')
        plt.clf()
        pass

    def run_episode(self, agent):
        self.reward = 0
        s = self.env.reset()
        done = False
        step=0
        r = 0
        actions = np.zeros(5)
        while not done:
            step+=1
            a = agent.act(s)
            if a == 0:
                actions[0] +=1
                r -= 1

            elif a == 1:
                actions[1] += 1
                r += 5

            elif a == 2:
                actions[2] += 1
                r += 5

            elif a == 3:
                actions[3] += 1
                r += 1

            elif a == 4:
                actions[4] += 1
                r += 1

            index, s_, price, gain, terminal, money = self.env.step(a)

            gain = gain if not terminal else 0

            if terminal:
                r -= 4000
                print("step: " + str(step) + " money: " +str(money), " rewards: " + str(r), " action", actions)
                self.steps.append(step)
                self.wallet.append(money)
                self.rewards.append(r)
                done = True

            elif step > 3300:
                if money > 3000:
                    r += 5000
                print("step: " + str(step) + " money: " + str(money)," rewards:"+ str(r), " action", actions)
                self.steps.append(step)
                self.wallet.append(money)
                self.rewards.append(r)
                done = True

            if gain > 0:
                r += 200
            if money>3000:
                r+=15
            r+=1
            agent.learn((s, a, s_, r, terminal))
            self.reward += r
            s = s_

        self.episode_count += 1
        self.reward_buffer.append(self.reward)
        average = sum(self.reward_buffer) / len(self.reward_buffer)

        print("Episode Nr. {} \nScore: {} \nAverage: {}".format(
            self.episode_count, self.reward, average))

if __name__ == "__main__":
    import gym
    import agent
    import observer
    data_path = '/home/jeffsd05/Bureau/final_project/Discrete/DDQN-master/data'
    exp = Experiment(data_path)
    agent = agent.DQNAgent(exp.env)
    epsilon = observer.EpsilonUpdater(agent)
    agent.add_observer(epsilon)
    exp.run_experiment(agent)

