# rl parameters
max_episodes = 100
max_step = 1000
capacity = 10000
replay_start_size = 1000

# noise parameters: Ornstein Uhlenbeck Process 
mu = 0
theta = 0.15
sigma = 0.2
dt = 1e-2

# neural network
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
batch_size = 128


# exploration
epsilon = 0.95
decay = 0.99

# discount
gamma = 0.99

# soft-update parameter
tau = 0.001
