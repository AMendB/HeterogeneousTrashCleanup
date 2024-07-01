import numpy as np
import torch

class ParticleSwarmOptimizationFleet:

    def __init__(self, env):

        # Get environment info #
        self.env = env
        self.n_agents = self.env.n_agents
        self.scenario_map = self.env.scenario_map.copy() 
        self.explorers_team_id = self.env.explorers_team_id
        self.cleaners_team_id = self.env.cleaners_team_id
        self.team_id_of_each_agent = self.env.team_id_of_each_agent
        self.movement_length_of_each_agent = self.env.movement_length_of_each_agent
        self.angle_set_of_each_agent = self.env.angle_set_of_each_agent
        self.vision_length_of_each_agent = self.env.vision_length_of_each_agent
        self.n_actions_of_each_agent = self.env.n_actions_of_each_agent
    
        # Constant ponderation values [c1, c2, c3, c4] https://arxiv.org/pdf/2211.15217.pdf #
        # c1 = best_local_locations, c2 = best_global_location, c3 = max_uncertainty, c4 = max_mean
        self.c_exploration = (2.0187, 0, 3.2697, 0) 
        self.c_explotation = (3.6845, 1.5614, 0, 3.6703) 
        self.c_cleaning = (0, 2, 0, 3)
        self.potential_movements_of_each_agent = {agent_id: np.array([(0,0) if angle < 0 else np.round([np.cos(angle), np.sin(angle)]) * self.movement_length_of_each_agent[agent_id] 
													 for angle in self.angle_set_of_each_agent[agent_id]]).astype(int) for agent_id in self.angle_set_of_each_agent.keys()}
        
        # Initialize the max measures and locations #
        self.max_trash_at_sight_per_agent = [0]*self.n_agents
        self.best_location_per_agent = [None]*self.n_agents
        self.velocities = [[0, 0]]*self.n_agents


        import matplotlib.pyplot as plt
        # Subplot to show the maps #
        _, self.axs = plt.subplots(1, 2, figsize=(10, 5))
        self.axs[0].set_title('Model Mean Map')
        self.axs[1].set_title('Model Uncertainty Map')

    def get_agents_actions(self):
        """ Update information and move the agent """

        self.active_agents_positions = self.env.get_active_agents_positions_dict()
        self.model_trash_map = self.env.model_trash_map.copy()
        self.visited_areas_map = self.env.visited_areas_map.copy()
        self.trash_locations = np.vstack(np.where(self.model_trash_map > 0)).T

        # Cleaners go for the trash since discovered, explorers explore the first half of the episode and then exploit #
        if self.env.steps > self.env.max_steps_per_episode/2 and len(self.trash_locations) > 0:
            self.c_values_of_each_agent = {agent_id: self.c_explotation if self.team_id_of_each_agent[agent_id] == self.explorers_team_id 
                                           else self.c_cleaning for agent_id in range(self.n_agents)}
        elif self.env.steps <= self.env.max_steps_per_episode/2 and len(self.trash_locations) > 0:
            self.c_values_of_each_agent = {agent_id: self.c_exploration if self.team_id_of_each_agent[agent_id] == self.explorers_team_id 
                                           else self.c_cleaning for agent_id in range(self.n_agents)}
        else:
            self.c_values_of_each_agent = {agent_id: self.c_exploration for agent_id in range(self.n_agents)}


        q_values = self.update_vectors()
        
        return q_values

    def update_vectors(self):
        """ Update the vectors direction of each agent """

        max_trash_location = np.unravel_index(np.argmax(self.model_trash_map), self.model_trash_map.shape)
        non_visited_locations = np.vstack(np.where(self.visited_areas_map == 1)).T # 1 is the value of non visited cells
        trash_at_sight_per_agent = [np.sum(self.env.real_trash_map[self.env.fleet.vehicles[agent_id].influence_mask.astype(bool)]) for agent_id in range(self.n_agents)]

        q_values = {}

        for agent_id, agent_position in self.active_agents_positions.items():

            # Update best historic local measure and its locations #
            if trash_at_sight_per_agent[agent_id] >= self.max_trash_at_sight_per_agent[agent_id]:
                self.max_trash_at_sight_per_agent[agent_id] = trash_at_sight_per_agent[agent_id]
                self.best_location_per_agent[agent_id] = agent_position

            # Get important locations #
            closer_unexplored_location = np.unravel_index(np.linalg.norm(np.array(agent_position) - non_visited_locations, axis=1).argmin(), self.scenario_map.shape)
            if len(self.trash_locations) > 0:
                closer_trash_location = np.unravel_index(np.linalg.norm(np.array(agent_position) - self.trash_locations, axis=1).argmin(), self.scenario_map.shape)

            # Update the vectors #
            vector_to_best_local_location = self.best_location_per_agent[agent_id] - agent_position
            vector_to_max_trash = max_trash_location - agent_position
            vector_to_closer_unexplored = closer_unexplored_location - agent_position
            vector_to_closer_trash = closer_trash_location - agent_position

            # Get final ponderated vector c*u #
            u_values = np.random.uniform(0, 1, 4) # random ponderation values
            vector = self.c_values_of_each_agent[agent_id][0] * u_values[0] * vector_to_best_local_location + \
                        self.c_values_of_each_agent[agent_id][1] * u_values[1] * vector_to_max_trash + \
                        self.c_values_of_each_agent[agent_id][2] * u_values[2] * vector_to_closer_unexplored + \
                        self.c_values_of_each_agent[agent_id][3] * u_values[3] * vector_to_closer_trash
            
            # Update the velocity of the agent #
            self.velocities[agent_id] = self.velocities[agent_id] + vector

            # Normalize the vector and get the movement direction #
            self.velocities[agent_id] = (self.velocities[agent_id] / np.linalg.norm(self.velocities[agent_id])) * self.movement_length_of_each_agent[agent_id]

            # Get Q-values in term of nearness to action #
            q_values[agent_id] = 1/(np.linalg.norm(self.potential_movements_of_each_agent[agent_id] - self.velocities[agent_id], axis=1) + 0.01) # 0.01 to avoid division by zero
            q_values[agent_id][8] = -10 # avoid staying in the same position

            # If cleaner and trash in pixel, q of the clean action is very high #
            if self.team_id_of_each_agent[agent_id] == self.cleaners_team_id and self.env.model_trash_map[agent_position[0], agent_position[1]] > 0:
                q_values[agent_id][9] = 100000
            elif self.team_id_of_each_agent[agent_id] == self.cleaners_team_id:
                q_values[agent_id][9] = -10 # avoid cleaning if there is no trash

        return q_values
    
    def reset(self):
        """ Reset the information of the algorithm """
    
        # Reset the max measures and locations #
        self.max_trash_at_sight_per_agent = [0]*self.n_agents
        self.best_location_per_agent = [None]*self.n_agents
        self.best_global_location = None
        self.velocities = [[0, 0]]*self.n_agents