from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
import numpy as np

class EnvWrapper:

    def __init__(self, 
                    scenario_map,
                    max_distance_travelled_by_team,
                    max_steps_per_episode,
                    number_of_agents_by_team,
                    n_actions_by_team,
                    fleet_initial_positions,
                    seed,
                    movement_length_by_team,
                    vision_length_by_team,
                    flag_to_check_collisions_within,
                    max_collisions,
                    reward_function,
                    dynamic, 
                    obstacles,
                    reward_weights,
                    show_plot_graphics,
                    ):
        
        self.env = MultiAgentCleanupEnvironment(scenario_map = scenario_map,
                                number_of_agents_by_team=number_of_agents_by_team,
                                n_actions_by_team=n_actions_by_team,
                                max_distance_travelled_by_team = max_distance_travelled_by_team,
                                max_steps_per_episode = max_steps_per_episode,
                                fleet_initial_positions = fleet_initial_positions, 
                                seed = seed,
                                movement_length_by_team = movement_length_by_team,
                                vision_length_by_team = vision_length_by_team,
                                flag_to_check_collisions_within = flag_to_check_collisions_within,
                                max_collisions = max_collisions,
                                reward_function = reward_function, 
                                reward_weights = reward_weights,
                                dynamic = dynamic,
                                obstacles = obstacles,
                                show_plot_graphics = show_plot_graphics,
                                )
        
        self.observation_space_shape = self.env.observation_space_shape
        self.max_steps_per_episode = self.env.max_steps_per_episode

    def reset(self):
        states = self.env.reset_env()
        states = np.array([states[i] for i in states.keys()])
        return states

    def step(self, actions):
        actions = {idx:action for idx, action in enumerate(actions)}
        states, rewards, done = self.env.step(actions)
        states = np.array([states[i] for i in states.keys()])
        rewards = np.array([*rewards.values()])
        done = np.array([*done.values()])
        return states, rewards, done

    def render(self):
        return self.env.render()
    
    def get_percentage_cleaned_trash(self):
        return self.env.get_percentage_cleaned_trash()
    
    def get_n_collisions(self):
        return self.env.fleet.fleet_collisions
    
    def save_environment_configuration(self, path):
        self.env.save_environment_configuration(path)