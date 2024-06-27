import numpy as np
from GaussianProcess.GPModels import GaussianProcessGPyTorch
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
        self.c_exploration = (2.0187, 0, 3.2697, 0) 
        self.c_explotation = (3.6845, 1.5614, 0, 3.6703) 
        self.potential_movements_of_each_agent = {agent_id: np.array([(0,0) if angle < 0 else np.round([np.cos(angle), np.sin(angle)]) * self.movement_length_of_each_agent[agent_id] 
													 for angle in self.angle_set_of_each_agent[agent_id]]).astype(int) for agent_id in self.angle_set_of_each_agent.keys()}
        
        # Initialize the max measures and locations #
        self.max_local_measures = [0]*self.n_agents
        self.max_global_measure = 0
        self.best_local_locations = [None]*self.n_agents
        self.best_global_location = None
        self.velocities = [[0, 0]]*self.n_agents


        # Declare Gaussian Process and models maps #
        self.model_mean_map = np.zeros_like(self.scenario_map) 
        self.model_uncertainty_map = np.zeros_like(self.scenario_map) 
        self.gaussian_process = GaussianProcessGPyTorch(scenario_map = self.scenario_map, 
                                                        initial_lengthscale = 1, kernel_bounds = (0.1, 3), 
                                                        training_iterations = 50, scale_kernel=True, 
                                                        device = 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the dictionaries that will store the vectors #
        self.vectors_to_best_local_location = {}
        self.vectors_to_best_global_location = {}
        self.vectors_to_max_uncertainty = {}
        self.vectors_to_max_mean = {}


        import matplotlib.pyplot as plt
        # Subplot to show the maps #
        _, self.axs = plt.subplots(1, 2, figsize=(10, 5))
        self.axs[0].set_title('Model Mean Map')
        self.axs[1].set_title('Model Uncertainty Map')

    def update_gaussian_process(self):
        """ The active agents take a sample from the real map as the sum of the 
        trashes in their vision range. It is used to update the Gaussian Process. """

        self.active_agents_positions = self.env.get_active_agents_positions_dict()
        position_measures = [*self.active_agents_positions.values()]
        self.measures = np.array([np.sum(self.env.real_trash_map[self.env.fleet.vehicles[agent_id].influence_mask.astype(bool)]) for agent_id in range(self.n_agents)])
        self.gaussian_process.fit_gp(X_new=position_measures , y_new=self.measures, variances_new=[0.0001]*len(self.measures))

        model_mean_map, model_uncertainty_map = self.gaussian_process.predict_gt()
        
        return model_mean_map, model_uncertainty_map

    def get_agents_actions(self):
        """ Update information and move the agent """
        self.model_mean_map, self.model_uncertainty_map = self.update_gaussian_process()

        self.axs[0].imshow(self.model_mean_map)
        self.axs[1].imshow(self.model_uncertainty_map)

        if self.env.steps > self.env.max_steps_per_episode:
            self.c_values = self.c_explotation
        else:
            self.c_values = self.c_exploration


        q_values = self.update_vectors()
        
        return q_values

    def update_vectors(self):
        """ Update the vectors direction of each agent """

        # Update the best global measure and its location #
        if np.max(self.measures) >= self.max_global_measure:
            self.max_global_measure = np.max(self.measures)
            self.best_global_location = self.active_agents_positions[np.argmax(self.measures)]

        # Update maximum of model mean and uncertainty location #
        self.max_mean_location = np.unravel_index(np.argmax(self.model_mean_map), self.model_mean_map.shape)
        self.max_uncertainty_location = np.unravel_index(np.argmax(self.model_uncertainty_map), self.model_uncertainty_map.shape)

        q_values = {}

        for agent_id, agent_position in self.active_agents_positions.items():

            # Update best historic local measure and its locations #
            if self.measures[agent_id] >= self.max_local_measures[agent_id]:
                self.max_local_measures[agent_id] = self.measures[agent_id]
                self.best_local_locations[agent_id] = agent_position

            # Update the vectors #
            self.vectors_to_best_local_location[agent_id] = self.best_local_locations[agent_id] - agent_position
            self.vectors_to_best_global_location[agent_id] = self.best_global_location - agent_position
            self.vectors_to_max_uncertainty[agent_id] = self.max_uncertainty_location - agent_position
            self.vectors_to_max_mean[agent_id] = self.max_mean_location - agent_position

            # Get final ponderated vector c*u #
            u_values = np.random.uniform(0, 1, 4) # random ponderation values
            vector = self.c_values[0] * u_values[0] * self.vectors_to_best_local_location[agent_id] + \
                        self.c_values[1] * u_values[1] * self.vectors_to_best_global_location[agent_id] + \
                        self.c_values[2] * u_values[2] * self.vectors_to_max_uncertainty[agent_id] + \
                        self.c_values[3] * u_values[3] * self.vectors_to_max_mean[agent_id]
            
            # Update the velocity of the agent #
            self.velocities[agent_id] = self.velocities[agent_id] + vector
            self.velocities[agent_id] = np.clip(self.velocities[agent_id], -self.movement_length_of_each_agent[agent_id], self.movement_length_of_each_agent[agent_id])

            # Normalize the vector #
            # movement = self.velocities[agent_id] / np.linalg.norm(self.velocities[agent_id])

            # Get Q-values in term of nearness to action #
            # q_values[agent_id] = 1/np.linalg.norm(self.potential_movements_of_each_agent[agent_id] - movement, axis=1)
            q_values[agent_id] = 1/np.linalg.norm(self.potential_movements_of_each_agent[agent_id] - self.velocities[agent_id], axis=1)

        return q_values
    
    def reset(self):
        """ Reset the information of the algorithm """
    
        # Reset the max measures and locations #
        self.max_local_measures = [0]*self.n_agents
        self.max_global_measure = 0
        self.best_local_locations = [None]*self.n_agents
        self.best_global_location = None
        self.velocities = [[0, 0]]*self.n_agents

        # Reset Gaussian Process and models maps #
        self.model_mean_map = np.zeros_like(self.scenario_map) 
        self.model_uncertainty_map = np.zeros_like(self.scenario_map) 
        self.gaussian_process.reset()

        # Reset the dictionaries that will store the vectors #
        self.vectors_to_best_local_location = {}
        self.vectors_to_best_global_location = {}
        self.vectors_to_max_uncertainty = {}
        self.vectors_to_max_mean = {}
