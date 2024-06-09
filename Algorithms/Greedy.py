import numpy as np

class OneStepGreedyFleet:
    """ Class to implement a 1-step greedy agent for the CleanupEnvironment. """

    def __init__(self, env) -> None:

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

        # Initialize variables #
        self.fixed_action = {agent_id: None for agent_id in range(self.n_agents)}

    def get_cleaner_action(self, agent_coords, agent_id):
        """ See the posibilities of the agent:

            - Case 0: If there is trash in the actual pixel, clean it.
            - Case 1: If there is trash in a accessible pixel, move to that pixel to clean it.
            - Case 2: If there is no trash in a accessible pixel, take a random action and continue until collision or detection of trash.
        
        If more than one trash pixel is at the same distance, select the one with more trash. """

        next_movements = np.array([(0,0) if angle < 0 else np.round([np.cos(angle), np.sin(angle)]) * self.movement_length_of_each_agent[agent_id] for angle in self.angle_set_of_each_agent[agent_id]]).astype(int)
        next_positions = agent_coords + next_movements
        next_positions = np.clip(next_positions, (0,0), np.array(self.navigable_map.shape)-1) # saturate movement if out of indexes values (map edges)
        number_of_trashes_in_each_next_position = np.array([self.model_trash_map[next_position[0], next_position[1]] if self.navigable_map[next_position[0], next_position[1]] == 1 
                                                            else -1 
                                                            for next_position in next_positions])

        if self.model_trash_map[agent_coords[0], agent_coords[1]] > 0: # Case 0
            action = 9
            self.fixed_action[agent_id] = None
        elif np.any(number_of_trashes_in_each_next_position > 0): # Case 1
            action = np.argmax(number_of_trashes_in_each_next_position) # Move to the pixel with more trash
            self.fixed_action[agent_id] = None
        # elif np.any(self.model_trash_map > 0): # Case 2
        #     # Get coords where there is trash
        #     trash_rows_cols = np.where(self.model_trash_map > 0)
        #     trash_coords = np.column_stack(trash_rows_cols)

        #     # Calculate distances to each trash pixel
        #     distances = np.linalg.norm(trash_coords - agent_coords, axis=0)

        #     # Get the closer pixel with more trash to the agent
        #     number_of_trashes_in_each_coord = self.model_trash_map[trash_coords]
        #     coord_to_clean =  np.lexsort((-number_of_trashes_in_each_coord, distances))[0]

        #     # Get the direction to move to the closer pixel with trash
        #     direction = trash_coords[coord_to_clean] - agent_coords
        #     desired_movement = np.round(direction / np.linalg.norm(direction)).astype(int) * self.movement_length_of_each_agent[agent_id]
        #     desired_next_position = agent_coords + desired_movement
        #     action = np.argmin(np.linalg.norm(possible_next_positions - desired_next_position, axis=1))
        else: # Case 3
            possible_actions = np.array([action_id for action_id, next_position in enumerate(next_positions) if self.navigable_map[next_position[0], next_position[1]] == 1 and np.any(next_position != agent_coords) ])
            
            if self.fixed_action[agent_id] in possible_actions:
                action = self.fixed_action[agent_id]
            else:
                action = np.random.choice(possible_actions)
                self.fixed_action[agent_id] = action
            
        self.navigable_map[next_positions[action][0], next_positions[action][1]] = 0 # to avoid collisions between agents

        return action

    def get_explorer_action(self, agent_coords, agent_id):
        """ See the posibilities of the agent:

            - Case 0: There is trash inside of its actual or next vision area. Then move to the area with more trash.
            - Case 1: There is no trash inside of its vision area. Then take an action that implies a movement and keep it until collision or detection of trash. """
            
        next_movements = np.array([(0,0) if angle < 0 else np.round([np.cos(angle), np.sin(angle)]) * self.movement_length_of_each_agent[agent_id] for angle in self.angle_set_of_each_agent[agent_id]]).astype(int)
        next_positions = agent_coords + next_movements
        next_positions = np.clip(next_positions, (0,0), np.array(self.navigable_map.shape)-1)

        # Calculate number of trashes in each next position influence area
        number_of_trashes_in_each_area = np.array([np.sum(self.model_trash_map[self.compute_influence_mask(next_position, self.vision_length_of_each_agent[agent_id]).astype(bool)]) 
                                                   if self.navigable_map[next_position[0], next_position[1]] == 1 
                                                   else -1 for next_position in next_positions])

        if np.any(number_of_trashes_in_each_area > 0): # Case 0
            action = np.argmax(number_of_trashes_in_each_area)
        else: # Case 1
            possible_actions = np.array([action_id for action_id, next_position in enumerate(next_positions) if self.navigable_map[next_position[0], next_position[1]] == 1 and np.any(next_position != agent_coords) ])
            
            if self.fixed_action[agent_id] in possible_actions:
                action = self.fixed_action[agent_id]
            else:
                action = np.random.choice(possible_actions)
                self.fixed_action[agent_id] = action
            
        self.navigable_map[next_positions[action][0], next_positions[action][1]] = 0 # to avoid collisions between agents
        
        return action

    def compute_influence_mask(self, agent_position, vision_length): 
        """ Compute influence area around actual position. It is what the agent can see. """

        influence_mask = np.zeros_like(self.scenario_map) 

        pose_x, pose_y = agent_position

        # State - coverage area #
        range_x_axis = np.arange(0, self.scenario_map.shape[0]) # posible positions in x-axis
        range_y_axis = np.arange(0, self.scenario_map.shape[1]) # posible positions in y-axis

        # Compute the circular mask (area) #
        mask = (range_x_axis[np.newaxis, :] - pose_x) ** 2 + (range_y_axis[:, np.newaxis] - pose_y) ** 2 <= vision_length ** 2 

        influence_mask[mask.T] = 1.0 # converts True values to 1 and False values to 0

        return influence_mask

    def get_agents_actions(self):
        """ Get the actions for each agent given the conditions of the environment. """
        
        self.navigable_map = self.scenario_map.copy() # 1 where navigable, 0 where not navigable
        self.model_trash_map = self.env.model_trash_map 
        active_agents_positions = self.env.get_active_agents_positions_dict()

        actions = {}

        for agent_id, agent_coords in active_agents_positions.items():
            if self.team_id_of_each_agent[agent_id] == self.cleaners_team_id:
                actions[agent_id] = self.get_cleaner_action(agent_coords, agent_id)
            elif self.team_id_of_each_agent[agent_id] == self.explorers_team_id:
                actions[agent_id] = self.get_explorer_action(agent_coords, agent_id)
        
        return actions
        
    def reset(self):
        """ Reset the variables associated to the decision making process. """

        self.fixed_action = {agent_id: None for agent_id in range(self.n_agents)}