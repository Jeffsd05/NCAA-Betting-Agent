import numpy as np
import bet_env
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

GAMMA = 1
LEARNING_RATE_set = [0.1]

MEMORY_SIZE = 1000000
BATCH_SIZE_set = [25, 50, 75, 100, 150]

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY_set = [0.99]



class DQNSolver:

    def __init__(self, observation_space, action_space, LEARNING_RATE):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self, BATCH_SIZE, exploration_rate):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate = exploration_rate
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def reset(self):
        self.memory = deque(maxlen=MEMORY_SIZE)


def betting_agent():
    env = bet_env.BetEnv('/home/jeffsd05/Bureau/final_project/Discrete/Bad Model - DQN/data')


    for BATCH_SIZE in BATCH_SIZE_set:
        for lr in LEARNING_RATE_set:
            for EXPLORATION_DECAY in EXPLORATION_DECAY_set:
                exploration_rate = EXPLORATION_MAX
                observation_space = env.observation_space.shape[0]+1
                action_space = env.action_space.n
                dqn_solver = DQNSolver(observation_space, action_space, lr)
                run = 0
                steps = []
                rewards = []
                reward = 0
                objectif = 3500
                wallet = []
                for i in np.arange(300):
                    run += 1
                    state = env.reset()
                    state = np.reshape(state, [1, observation_space])
                    step = 0
                    actions = np.zeros(5)

                    while True:
                        step += 1
                        action = dqn_solver.act(state)

                        if action==0:
                            reward -=100
                            actions[0]+=1
                        elif action==1:
                            actions[1]+=1
                            reward += 50
                        elif action==2:
                            actions[2]+=1
                            reward += 50
                        elif action==3:
                            actions[3] += 1
                            reward += 1
                        elif action==4:
                            actions[4] += 1
                            reward += 1

                        index, state_next, price, gain, terminal, money = env.step(action)
                        gain = gain if not terminal else 0
                        if gain > 0:
                            reward += 200


                        if terminal:
                            reward -= 1000
                            steps.append(step)
                            wallet.append(money)
                            exploration_rate *=  EXPLORATION_DECAY
                            break

                        elif step > 3300:
                            if money > 3000:
                                reward+=1000
                            reward += 2000
                           
                            rewards.append(reward)
                            steps.append(step)
                            wallet.append(money)
                            exploration_rate *= EXPLORATION_DECAY
                            break


                        state_next = np.reshape(state_next, [1, observation_space])
                        dqn_solver.remember(state, action, reward, state_next, terminal)
                        state = state_next
                        dqn_solver.experience_replay(BATCH_SIZE, exploration_rate)


                plt.plot(steps)
                plt.xlabel('Number of episode')
                plt.ylabel('Number of step before brankuptcy')
                plt.title('DQN learning curve with batch size: '+str(BATCH_SIZE)+',\n learning rate: '+ str(lr) +\
                          ' and exploration Decay: '+ str(EXPLORATION_DECAY))
                plt.savefig('steps lr_'+str(lr)+'BS_'+str(BATCH_SIZE)+'ED_'+str(EXPLORATION_DECAY)+'.png')
                plt.clf()

                plt.plot(rewards)
                plt.xlabel('Number of episode')
                plt.ylabel('Rewards after episode')
                plt.title('DQN learning curve with batch size: ' + str(BATCH_SIZE) + ',\n learning rate: ' + str(lr) + \
                          ' and exploration Decay: ' + str(EXPLORATION_DECAY))
                plt.savefig('rewards lr_' + str(lr) + 'BS_' + str(BATCH_SIZE) + 'ED_' + str(EXPLORATION_DECAY) + '.png')
                plt.clf()

                plt.plot(wallet)
                plt.xlabel('Number of episode')
                plt.ylabel('Rewards after episode')
                plt.title('DQN learning curve with batch size: ' + str(BATCH_SIZE) + ',\n learning rate: ' + str(lr) + \
                          ' and exploration Decay: ' + str(EXPLORATION_DECAY))
                plt.savefig('money lr_' + str(lr) + 'BS_' + str(BATCH_SIZE) + 'ED_' + str(EXPLORATION_DECAY) + '.png')
                plt.clf()


if __name__ == "__main__":
  betting_agent()


