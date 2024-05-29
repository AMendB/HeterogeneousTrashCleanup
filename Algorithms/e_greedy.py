import numpy as np

class EGreedyAgent:
    def __init__(self, env, rw_fn, rw_weights) -> None:
        self.env = env
        self.reward_function = rw_fn
        self.reward_weights = rw_weights
        self.epsilon = 0.1
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        # Import get_reward function from CleanupEnvironment.py
        self.get_reward = self.env.get_reward
        
    def move(self, state):
        pass
        