import numpy as np

class EGreedyAgent:
    """ Class to implement a 1-step epsilon greedy agent for the CleanupEnvironment. """
    def __init__(self, env, rw_fn, rw_weights) -> None:
        self.env = env
        self.reward_function = rw_fn
        self.reward_weights = rw_weights
        self.epsilon = 0.1
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        # Import get_reward function from CleanupEnvironment.py
        self.get_reward = self.env.get_reward

        self.explorers_team_id = self.env.explorers_team_id
        self.cleaners_team_id = self.env.cleaners_team_id
        self.team_id_of_each_agent = self.env.team_id_of_each_agent
    
    def get_closer_trash(self, pose):
        # Get the closer pixel with trash to the agent
        trash_pixels = np.where(self.env.scenario_map == 2)
        distances = np.linalg.norm(np.array(trash_pixels) - np.array(pose), axis=0)
        closer_trash = trash_pixels[np.argmin(distances)]
        return closer_trash

    def move(self, id):
        if id in self.explorers_team_id:
            pass
        elif id in self.cleaners_team_id:
            pose = self.env.fleet.vehicles[id].actual_agent_position
            # Get the closer pixel with trash to the agent
            distances_to_trash = np.linalg.norm(np.array(self.env.model_trash_map) - np.array(pose), axis=0)
            # If more than 1 trash pixel is at the same distance, select the one with more trash
            closer_trash = np.array(np.where(distances_to_trash == np.min(distances_to_trash)))
        