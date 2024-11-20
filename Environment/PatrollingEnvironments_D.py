import sys
sys.path.append('.')
import gym
import numpy as np
import matplotlib.pyplot as plt
# from Environment.GroundTruthsModels.MacroPlasticGroundtruth import macro_plastic, macroplastic_colormap, background_colormap
# from Environment.Wrappers.time_stacking_wrapper import MultiAgentTimeStackingMemory
from Environment.GroundTruthsModels.MacroPlasticGroundTruth import macro_plastic, macroplastic_colormap, background_colormap
from Environment.Wrappers.time_stacking_wrapper import MultiAgentTimeStackingMemory
from scipy.spatial import distance_matrix
import matplotlib
import json
from collections import deque
from scipy.ndimage import gaussian_filter
import heapq

background_colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["sienna","dodgerblue"])

class DiscreteVehicle:

	def __init__(self, initial_position, n_actions, movement_length, navigation_map, detection_length):
		
		""" Initial positions of the drones """
		self.initial_position = initial_position
		self.position = np.copy(initial_position)

		""" Initialize the waypoints """
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)

		""" Detection radius for the contmaination vision """
		self.detection_length = detection_length
		self.navigation_map = navigation_map
		self.detection_mask = self.compute_detection_mask()

		""" Reset other variables """
		self.distance = 0.0
		self.num_of_collisions = 0
		self.action_space = gym.spaces.Discrete(n_actions)
		self.angle_set = np.linspace(0, 2 * np.pi, n_actions, endpoint=False)
		self.movement_length = movement_length

		

	def move(self, action, valid=True):
		""" Move a vehicle in the direction of the action. If valid is False, the action is not performed. """
		if action < len(self.angle_set):
			angle = self.angle_set[action]
			movement = np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])).astype(int)
			next_position = self.position + movement
			self.distance += np.linalg.norm(self.position - next_position)
		else:
			next_position = self.position
			self.distance += self.movement_length

		if self.check_collision(next_position) or not valid:
			collide = True
			self.num_of_collisions += 1
		else:
			collide = False
			self.position = next_position
			self.waypoints = np.vstack((self.waypoints, [self.position]))

		self.detection_mask = self.compute_detection_mask()

		return collide

	def check_collision(self, next_position):
		if (next_position[0] < 0) or (next_position[0] >= self.navigation_map.shape[0]) or (next_position[1] < 0) or (next_position[1] >= self.navigation_map.shape[1]):
			return True
		if self.navigation_map[int(next_position[0]), int(next_position[1])] == 0:
			return True  # There is a collision

		return not self.is_reachable(next_position)

	def is_reachable(self, next_position):
		""" Check if the next position is reachable """
		x, y = next_position
		dx = x - self.position[0]
		dy = y - self.position[1]
		steps = max(abs(dx), abs(dy))
		dx = dx / steps if steps != 0 else 0
		dy = dy / steps if steps != 0 else 0
		reachable_positions = True
		for step in range(1, steps + 1):
			px = round(self.position[0] + dx * step)
			py = round(self.position[1] + dy * step)
			if self.navigation_map[px, py] != 1:
				reachable_positions = False
				break

		return reachable_positions

	def compute_detection_mask(self):
		""" Compute the circular mask """

		known_mask = np.zeros_like(self.navigation_map)

		px, py = self.position.astype(int)

		# State - coverage area #
		x = np.arange(0, self.navigation_map.shape[0])
		y = np.arange(0, self.navigation_map.shape[1])

		# Compute the circular mask (area) of the state 3 #
		mask = (x[np.newaxis, :] - px) ** 2 + (y[:, np.newaxis] - py) ** 2 <= self.detection_length ** 2

		known_mask[mask.T] = 1.0
		known_mask = known_mask * self.navigation_map
		for px, py in np.argwhere(known_mask == 1):
			if not self.is_reachable([px, py]):
				known_mask[px, py] = 0
		return known_mask*self.navigation_map

	def reset(self, initial_position):
		""" Reset the agent - Position, detection mask, etc. """

		self.initial_position = initial_position
		self.position = np.copy(initial_position)
		self.waypoints = np.expand_dims(np.copy(initial_position), 0)
		self.distance = 0.0
		self.num_of_collisions = 0
		self.detection_mask = self.compute_detection_mask()

	def check_action(self, action):
		""" Return True if the action leads to a collision """
		if action >= len(self.angle_set):
			return False
		angle = self.angle_set[action]
		movement = np.round(np.array([self.movement_length * np.cos(angle), self.movement_length * np.sin(angle)])).astype(int)
		next_position = self.position + movement

		return self.check_collision(next_position)

	def move_to_position(self, goal_position):
		""" Move to the given position """
		assert (goal_position[0] > 0) or (goal_position[0] < self.navigation_map.shape[0]) or (goal_position[1] > 0) or (goal_position[1] < self.navigation_map.shape[1]) , "Invalid position to move"

		assert self.navigation_map[goal_position[0], goal_position[1]] == 1, "Invalid position to move"
		self.distance += np.linalg.norm(goal_position - self.position)
		""" Update the position """
		self.position = goal_position

class DiscreteFleet:

	def __init__(self,
				 number_of_vehicles,
				 n_actions,
				 fleet_initial_positions,
				 movement_length,
				 detection_length,
				 navigation_map):

		""" Coordinator of the movements of the fleet. Coordinates the common model, the distance between drones, etc. """
		self.number_of_vehicles = number_of_vehicles
		self.initial_positions = fleet_initial_positions
		self.n_actions = n_actions
		self.movement_length = movement_length
		self.detection_length = detection_length

		""" Create the vehicles object array """
		self.vehicles = [DiscreteVehicle(initial_position=fleet_initial_positions[k],
										 n_actions=n_actions,
										 movement_length=movement_length,
										 navigation_map=navigation_map,
										 detection_length=detection_length) for k in range(self.number_of_vehicles)]

		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])

		# Get the redundancy mask #
		self.redundancy_mask = np.sum([veh.detection_mask for veh in self.vehicles], axis=0)
		# Get the collective detection mask #
		self.collective_mask = self.redundancy_mask.astype(bool)
		self.historic_visited_mask = self.redundancy_mask.astype(bool)
		self.new_visited_mask = self.historic_visited_mask
		self.fleet_collisions = 0

	def check_fleet_collision_within(self, veh_actions):
		""" Check if there is any collision between agents """
		
		new_positions = []

		for idx, veh_action in veh_actions.items():
			if veh_action >= len(self.vehicles[idx].angle_set):
				new_positions.append(list(self.vehicles[idx].position))
			else:
				angle = self.vehicles[idx].angle_set[veh_action]
				movement = np.round(np.array([self.vehicles[idx].movement_length * np.cos(angle), self.vehicles[idx].movement_length * np.sin(angle)])).astype(int)
				new_positions.append(list(self.vehicles[idx].position + movement))

		_, inverse_index, counts = np.unique(np.asarray(new_positions), return_inverse=True, return_counts=True, axis=0)

		# True if repeated #
		not_collision_within = counts[inverse_index] == 1
		return not_collision_within

	def move(self, fleet_actions):

		# Check if there are collisions between vehicles #
		self_colliding_mask = self.check_fleet_collision_within(fleet_actions)  ## We should add True so these collisions doesn't affect
		# Process the fleet actions and move the vehicles #
		collision_array = {k: self.vehicles[k].move(fleet_actions[k], valid=valid) for k, valid in zip(list(fleet_actions.keys()), self_colliding_mask)}
		# Update vector with agent positions #
		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])
		# Sum up the collisions for termination #
		self.fleet_collisions = np.sum([self.vehicles[k].num_of_collisions for k in range(self.number_of_vehicles)])
		# Compute the redundancy mask #
		self.redundancy_mask = np.sum([self.vehicles[agent_id].detection_mask for agent_id in fleet_actions.keys()], axis=0)
		# Update the collective mask #
		self.collective_mask = self.redundancy_mask.astype(bool)
		# Update the historic visited mask #
		previous_historic_visited_mask = self.historic_visited_mask
		self.historic_visited_mask = np.logical_or(self.historic_visited_mask, self.collective_mask)
		self.new_visited_mask = np.logical_xor(self.historic_visited_mask, previous_historic_visited_mask)
		return collision_array


	def reset(self, initial_positions=None):
		""" Reset the fleet """

		if initial_positions is None:
			initial_positions = self.initial_positions

		for k in range(self.number_of_vehicles):
			self.vehicles[k].reset(initial_position=initial_positions[k])

		self.agent_positions = np.asarray([veh.position for veh in self.vehicles])

		self.fleet_collisions = 0

		# Get the redundancy mask #
		self.redundancy_mask = np.sum([veh.detection_mask for veh in self.vehicles], axis=0)
		# Get the collective detection mask #
		self.collective_mask = self.redundancy_mask.astype(bool)
		self.historic_visited_mask = self.redundancy_mask.astype(bool)
		self.new_visited_mask = self.historic_visited_mask

	def get_distances(self):
		return [self.vehicles[k].distance for k in range(self.number_of_vehicles)]

	def check_collisions(self, test_actions):
		""" Array of bools (True if collision) """
		return [self.vehicles[k].check_action(test_actions[k]) for k in range(self.number_of_vehicles)]

	def move_fleet_to_positions(self, goal_list):
		""" Move the fleet to the given positions.
		 All goal positions must ve valid. """

		goal_list = np.atleast_2d(goal_list)

		for k in range(self.number_of_vehicles):
			self.vehicles[k].move_to_position(goal_position=goal_list[k])

	def get_distance_matrix(self):
		return distance_matrix(self.agent_positions, self.agent_positions)

	def get_positions(self):

		return np.asarray([veh.position for veh in self.vehicles])


class MultiAgentPatrolling(gym.Env):

	def __init__(self, scenario_map,
				 distance_budget,
				 number_of_vehicles,
				 fleet_initial_positions=None,
				 seed=0,
				 miopic=True,
				 dynamic=True,
				 detection_length=2,
				 movement_length=2,
				 max_collisions=5,
				 obstacles=False,
				 reward_type='Double reward',
				 ground_truth_type='macro_plastic',
				 convert_to_uint8=True,
				 frame_stacking = 0,
				 state_index_stacking = (0,1,2,3,4),
     			 trail_length = 10	):

		""" The gym environment """

		# Load the scenario map
		self.scenario_map = scenario_map
		self.visitable_locations = np.vstack(np.where(self.scenario_map != 0)).T
		self.number_of_agents = number_of_vehicles
		self.action_space = gym.spaces.Discrete(16)
		# Graph and distance maps
		self.graph = self.grid_to_graph()
		self.distance_map, self.predecessor_map = self.calculate_distance_and_predecessor_maps()
  
		# Initial positions
		if fleet_initial_positions is None:
			self.random_inititial_positions = True
			self.rng_initial_positions = np.random.default_rng(seed)
			random_positions_indx = self.rng_initial_positions.choice(np.arange(0, len(self.visitable_locations)), number_of_vehicles, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]
		else:
			self.random_inititial_positions = False
			self.initial_positions = fleet_initial_positions

		self.obstacles = obstacles
		self.miopic = miopic
		if self.obstacles:
			self.rng_obstacles = np.random.default_rng(seed)
		self.reward_type = reward_type
	
		# Number of pixels
		self.distance_budget = distance_budget
		self.min_movements_if_nocollisions = distance_budget // detection_length
		# Number of agents
		self.seed = seed
		# Detection radius
		self.detection_length = detection_length
		# Fleet of N vehicles
		self.movement_length = movement_length
		
		# Create the fleets 
		self.fleet = DiscreteFleet(number_of_vehicles=self.number_of_agents,
								   n_actions=8,
								   fleet_initial_positions=self.initial_positions,
								   movement_length=movement_length,
								   detection_length=detection_length,
								   navigation_map=self.scenario_map)

		self.max_collisions = max_collisions
		# Ground truth
		self.dynamic = dynamic
		self.ground_truth_type = ground_truth_type
		if ground_truth_type == 'macro_plastic':
			self.gt = macro_plastic(self.scenario_map, seed=self.seed)
		else:
			raise NotImplementedError("This Benchmark is not implemented. Choose one that is.")
		

		""" Model attributes """
		self.known_information = None
		self.normalized_known_information = None
		self.macro_plastic_gt = None
		self.model = None
		self.inside_obstacles_map = None
		self.state = None
		self.fig = None
		self.convert_to_uint8 = convert_to_uint8
		assert frame_stacking >= 0, "frame_stacking must be >= 0"
		self.state_index_stacking = state_index_stacking
		self.num_of_frame_stacking = frame_stacking
		self.n_channels = 3
		if frame_stacking != 0:
			self.frame_stacking = MultiAgentTimeStackingMemory(n_agents = self.number_of_agents,
			 													n_timesteps = frame_stacking - 1, 
																state_indexes = state_index_stacking, 
																n_channels = self.n_channels)
			self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_channels + len(state_index_stacking)*(frame_stacking - 1), *self.scenario_map.shape), dtype=np.float32)

		else:
			self.frame_stacking = None
			self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_channels, *self.scenario_map.shape), dtype=np.float32)

		self.state_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_channels, *self.scenario_map.shape), dtype=np.float32)

		# Trail
		self.trail_length = trail_length
		self.last_positions = [deque(maxlen=self.trail_length) for _ in range(self.number_of_agents)]
		# Metrics
		self.steps = 0
		self.total_n_trash_cleaned = np.zeros(self.number_of_agents)
		self.min_percentage_of_trash_found = 0
		self.percentage_of_trash_cleaned = 0
		self.percentage_of_map_visited = 0
  
	def reset(self):
		""" Reset the environment """

		# Reset the ground truth #
		self.gt.reset()
		self.macro_plastic_gt = self.gt.read()
		# Create an empty model #
		self.model = np.zeros_like(self.scenario_map) if self.miopic else self.macro_plastic_gt
		self.model_ant = self.model.copy()

		# Get the N random initial positions #
		if self.random_inititial_positions:
			random_positions_indx = self.rng_initial_positions.choice(np.arange(0, len(self.visitable_locations)), self.number_of_agents, replace=False)
			self.initial_positions = self.visitable_locations[random_positions_indx]

		# Reset the positions of the fleet #
		self.fleet.reset(initial_positions=self.initial_positions)
		self.active_agents = {agent_id: True for agent_id in range(self.number_of_agents)}

		# Randomly generated obstacles #
		if self.obstacles:
			# Generate a inside obstacles map #
			self.inside_obstacles_map = np.zeros_like(self.scenario_map)
			obstacles_pos_indx = self.rng_obstacles.choice(np.arange(0, len(self.visitable_locations)), size=20, replace=False)
			self.inside_obstacles_map[self.visitable_locations[obstacles_pos_indx, 0], self.visitable_locations[obstacles_pos_indx, 1]] = 1.0

			# Update the obstacle map for every agent #
			for i in range(self.number_of_agents):
				self.fleet.vehicles[i].navigation_map = self.scenario_map - self.inside_obstacles_map
    
		self.last_positions = [deque(maxlen=self.trail_length) for _ in range(self.number_of_agents)]
		
		self.n_trash_cleaned = np.array([0 for _ in range(self.number_of_agents)])
		self.no_discovery_steps = np.zeros(self.number_of_agents)
		# Update the state of the agents #
		self.update_state()
		# Metrics
		self.steps = 0
		self.total_n_trash_cleaned = np.zeros(self.number_of_agents)
		self.min_percentage_of_trash_found = 0
		self.percentage_of_trash_cleaned = 0
		self.percentage_of_map_visited = 0
		self.n_of_trash = np.sum(self.gt.number_of_trash_elements_in_each_spot)	
		self.update_metrics()
    
		return self.state if self.frame_stacking is None else self.frame_stacking.process(self.state)


	def update_state(self):
		""" Update the state for every vehicle """

		state = {}

		# State 0 -> Known boundaries
		if self.obstacles:
			obstacle_map = self.scenario_map - self.inside_obstacles_map
		else:
			obstacle_map = self.scenario_map

		# State 2 -> Known information
		# state[2] = self.macro_plastic_gt * self.fleet.historic_visited_mask if self.miopic else self.macro_plastic_gt
		if self.miopic:
			self.known_information = np.zeros_like(self.scenario_map)
			self.known_information[np.where(self.fleet.historic_visited_mask)] = self.model[np.where(self.fleet.historic_visited_mask)]
		else:
			self.known_information = self.gt.read()
		
		# Normalize
		if np.max(self.known_information) == 0:
			self.normalized_known_information = np.zeros_like(self.known_information)
		else:
			self.normalized_known_information = (self.known_information-np.min(self.known_information))/(np.max(self.known_information)-np.min(self.known_information))

		for i in range(self.number_of_agents):
	
			agent_observation_of_position = self.fleet.vehicles[i].detection_mask.copy()

			self.last_positions[i].append(agent_observation_of_position.copy())
			trail_length = len(self.last_positions[i])
			trail_values = np.linspace(1,0,trail_length, endpoint=False)
			for j, pos in enumerate(self.last_positions[i]):
				agent_observation_of_position[pos.astype(bool)] = np.flip(trail_values)[j]	
	
			agent_observation_of_fleet = self.fleet.redundancy_mask.copy() - self.fleet.vehicles[i].detection_mask.copy()
			
			# Set cells that are 0 in self.scenario_map to -1 in the observations
			# self.normalized_known_information[self.scenario_map == 0] = -1
			# agent_observation_of_position[self.scenario_map == 0] = -1
			# agent_observation_of_fleet[self.scenario_map == 0] = -1
   
			state[i] = np.concatenate((
				self.normalized_known_information[np.newaxis],
				agent_observation_of_position[np.newaxis],
				agent_observation_of_fleet[np.newaxis]
			))
			if self.convert_to_uint8:
			# Convert the state to uint8
				state[i] = np.round(state[i] * 255).astype(np.uint8)
		self.state = {agent_id: state[agent_id] for agent_id in range(self.number_of_agents) if self.active_agents[agent_id]}

	def step(self, action: dict):

		# Process action movement only for active agents #
		action = {action_id: action[action_id] for action_id in range(self.number_of_agents) if self.active_agents[action_id]}
		collision_mask = self.fleet.move(action)
		self.n_trash_cleaned = np.array([0 for _ in range(self.number_of_agents)])
		# Clean the trash if requested # --> Changed this to clean the trash if the agent is in a trash spot
		
		for agent_id in range(self.number_of_agents):
			if self.active_agents[agent_id]:
				#if action[agent_id] == 8:
				n_trash_to_clean = 100
				#n_trash_to_clean = 26 -  self.total_n_trash_cleaned[agent_id]
				if self.gt.map[self.fleet.vehicles[agent_id].position[0],self.fleet.vehicles[agent_id].position[1]] > 0:
					self.n_trash_cleaned[agent_id] = self.gt.clean_particles(self.fleet.vehicles[agent_id].position, n_trash_to_clean)
       # Update model #
		if self.miopic:
			self.update_model(action)
		else:
			self.model = self.gt.read()

		# Compute reward
		reward = self.reward_function(collision_mask, action)
		self.macro_plastic_gt = self.gt.read()

		# Update state
		self.update_state()

		# Update metrics
		self.steps += 1
		self.update_metrics()
  
		# Final condition #
		done = {agent_id: self.fleet.get_distances()[agent_id] > self.distance_budget 
          		or self.fleet.fleet_collisions > self.max_collisions 
            	or self.percentage_of_trash_cleaned == 1
				#or self.total_n_trash_cleaned[agent_id] > 25
                for agent_id in range(self.number_of_agents)}
		self.active_agents = [not d for d in done.values()]
		
		# Update ground truth if dynamic #
		if self.dynamic:
			self.gt.step()

		return self.state if self.frame_stacking is None else self.frame_stacking.process(self.state), reward, done, self.info

	def update_model(self,action):
		""" Update the model using the new positions """

		self.model_ant = self.model.copy()

		gt_ = self.gt.read()
		for idx, vehicle in enumerate(self.fleet.vehicles):
			if self.active_agents[idx]:
				self.model[vehicle.detection_mask.astype(bool)] = gt_[vehicle.detection_mask.astype(bool)]
    
	def update_metrics(self):
		""" Update the metrics """
		self.total_n_trash_cleaned += self.n_trash_cleaned
		self.percentage_of_trash_cleaned = np.sum(self.total_n_trash_cleaned) / self.n_of_trash
		self.min_percentage_of_trash_found = max(self.min_percentage_of_trash_found,
                                              np.sum(self.gt.map[self.fleet.historic_visited_mask.astype(bool)] > 0) / self.n_of_trash)
		self.percentage_of_map_visited = np.sum(self.fleet.historic_visited_mask) / np.sum(self.scenario_map)

    
	def render(self):

		import matplotlib.pyplot as plt

		agente_disponible = np.argmax(self.active_agents)

		if not any(self.active_agents):
			return

		if self.convert_to_uint8:
			vmin = 0
			vmax = 255.0
		else:
			vmin = 0.0
			vmax = 1.0
		if self.fig is None:

			self.fig, self.axs = plt.subplots(1, 6, figsize=(15,5))
			# Print the obstacles map
			self.im0 = self.axs[0].imshow(self.scenario_map, cmap = background_colormap)
			self.axs[0].set_title('Navigation map')
			# Print the ground truth
			real_gt = self.scenario_map*np.nan
			real_gt[self.visitable_locations[:,0], self.visitable_locations[:,1]] = self.macro_plastic_gt[self.visitable_locations[:,0], self.visitable_locations[:,1]]
			self.im1 = self.axs[1].imshow(real_gt,  cmap=macroplastic_colormap, vmin=vmin)
			self.axs[1].set_title("Real importance GT")

			# Print model  #
			model_gt = self.scenario_map*np.nan
			model_gt[self.visitable_locations[:,0], self.visitable_locations[:,1]] = self.state[agente_disponible][0][self.visitable_locations[:,0], self.visitable_locations[:,1]]
			
			self.im2 = self.axs[2].imshow(model_gt, cmap=macroplastic_colormap,vmin=vmin,vmax=self.macro_plastic_gt.max())
			self.axs[2].set_title("Model")
   
			# Agent 0 position #
			self.im3 = self.axs[3].imshow(self.state[agente_disponible][1], cmap = 'gray')
			self.axs[3].set_title("Agent 0 position")

			# Others-than-Agent 0 position #
			self.im4 = self.axs[4].imshow(self.state[agente_disponible][2], cmap = 'gray')
			self.axs[4].set_title("Others agents position")
			# Redundacy
			
			self.im5 = self.axs[5].imshow(self.fleet.historic_visited_mask, cmap = 'gray')
			self.axs[5].set_title("Redundacy Mask")

		self.im0.set_data(self.scenario_map)
  
		real_gt = self.scenario_map*np.nan
		real_gt[self.visitable_locations[:,0], self.visitable_locations[:,1]] = self.macro_plastic_gt[self.visitable_locations[:,0], self.visitable_locations[:,1]]
		self.im1.set_data(real_gt)
		model_gt = self.scenario_map*np.nan
		model_gt[self.visitable_locations[:,0], self.visitable_locations[:,1]] = self.state[agente_disponible][0][self.visitable_locations[:,0], self.visitable_locations[:,1]]
		
		self.im2.set_data(model_gt)
		self.im3.set_data(self.state[agente_disponible][1])
		self.im4.set_data(self.state[agente_disponible][2])
		self.im5.set_data(self.fleet.historic_visited_mask)

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

		plt.draw()

		plt.pause(0.01)

	def reward_function(self, collision_mask, actions):
		
		trash_monitoring_reward = np.array([np.sum( np.clip(self.gt.map[veh.detection_mask.astype(bool)],0,1)
				/ (np.sum(veh.detection_mask) * self.fleet.redundancy_mask[veh.detection_mask.astype(bool)]))
					for veh in self.fleet.vehicles])
		if 'Double reward' in self.reward_type:
			""" Compute the reward for the agents """
			rewards_exploration = trash_monitoring_reward + np.array(
				[np.sum(self.fleet.new_visited_mask[veh.detection_mask.astype(bool)].astype(np.float32) 
						/ self.fleet.redundancy_mask[veh.detection_mask.astype(bool)]) for veh in self.fleet.vehicles]
			)
			#rewards_cleaning = trash_monitoring_reward + self.n_trash_cleaned 
			"""rewards_cleaning = trash_monitoring_reward + self.n_trash_cleaned + np.array([
						np.sum(self.gt.normalized_filtered_map[veh.detection_mask.astype(bool)]
								/ (np.sum(veh.detection_mask) * self.fleet.redundancy_mask[veh.detection_mask.astype(bool)]))
								for veh in self.fleet.vehicles])"""
			rewards_cleaning = self.n_trash_cleaned + np.array([
						np.sum(self.gt.normalized_filtered_map[veh.detection_mask.astype(bool)]
								/ (np.sum(veh.detection_mask) * self.fleet.redundancy_mask[veh.detection_mask.astype(bool)]))
								for veh in self.fleet.vehicles])
		if 'Negative reward' in  self.reward_type:
			rewards_exploration = trash_monitoring_reward + np.array(
			[np.sum(self.fleet.new_visited_mask[veh.detection_mask.astype(bool)].astype(np.float32) 
                    / self.fleet.redundancy_mask[veh.detection_mask.astype(bool)]) for veh in self.fleet.vehicles])
   
			# if an agent has spent a step without discovering anything, 
   			# it will receive a negative reward that will increase over time, 
      		# it will be reset when it discovers something	
			for idx, agent in enumerate(self.fleet.vehicles):
				if self.percentage_of_map_visited == 1:
					continue
				if self.active_agents[idx]:
					if np.sum(self.fleet.new_visited_mask[agent.detection_mask.astype(bool)]) == 0:
						self.no_discovery_steps[idx] += 1
					else:
						self.no_discovery_steps[idx] = 0
					rewards_exploration[idx] -= (1/self.min_movements_if_nocollisions) * self.no_discovery_steps[idx]
			
			filtered_map = gaussian_filter(self.normalized_known_information, 5, mode = 'constant', cval=0, radius = None) * self.scenario_map
			filtered_map = (filtered_map - np.min(filtered_map)) / (np.max(filtered_map) - np.min(filtered_map) + 1e-8)
			if self.percentage_of_trash_cleaned == 1 and np.sum(filtered_map)==0:
				rewards_cleaning = np.ones_like(rewards_exploration)
			else:
			#model_change = np.array([np.sum(self.gt.map[veh.detection_mask.astype(bool)] - 
				rewards_cleaning = self.n_trash_cleaned - (1 - np.array([
							np.sum(filtered_map[veh.detection_mask.astype(bool)]
									/ (np.sum(veh.detection_mask) * self.fleet.redundancy_mask[veh.detection_mask.astype(bool)]))
								for i,veh in enumerate(self.fleet.vehicles)]))
			# when model change, the agent will receive a positive reward
   
			if self.model_ant is not None:
				model_change = np.array([np.sum(np.abs(self.model[veh.detection_mask.astype(bool)] - self.model_ant[veh.detection_mask.astype(bool)])) for veh in self.fleet.vehicles])
				rewards_cleaning += model_change
		if 'Distance Field' in self.reward_type:
			rewards_exploration = np.array(
			[np.sum(self.fleet.new_visited_mask[veh.detection_mask.astype(bool)].astype(np.float32) 
                    / self.fleet.redundancy_mask[veh.detection_mask.astype(bool)]) for veh in self.fleet.vehicles])
   
			# if an agent has spent a step without discovering anything, 
   			# it will receive a negative reward that will increase over time, 
      		# it will be reset when it discovers something	
			for idx, agent in enumerate(self.fleet.vehicles):
				if self.percentage_of_map_visited == 1:
					continue
				if self.active_agents[idx]:
					if np.sum(self.fleet.new_visited_mask[agent.detection_mask.astype(bool)]) == 0:
						self.no_discovery_steps[idx] += 1
						# Penalize based on the number of last positions in the detection mask
						penalty = 0	
						for i in range(len(self.last_positions)):
							penalty += sum([np.sum(pos.astype(bool) & agent.detection_mask.astype(bool)) 
                     					for pos in self.last_positions[i]])
						rewards_exploration[idx] -= penalty/(np.sum(agent.detection_mask)*10)
					else:
						self.no_discovery_steps[idx] = 0
					rewards_exploration[idx] -= (1/self.min_movements_if_nocollisions) * self.no_discovery_steps[idx]
					#rewards_exploration[idx] -= 1
			filtered_map = self.calculate_field_map(self.normalized_known_information, self.fleet.collective_mask, alpha=1.0)
			#filtered_map = (filtered_map - np.min(filtered_map)) / (np.max(filtered_map) - np.min(filtered_map) + 1e-8)	
			if self.percentage_of_trash_cleaned == 1 and np.sum(filtered_map)==0:
				rewards_cleaning = np.ones_like(rewards_exploration)
			else:
			#model_change = np.array([np.sum(self.gt.map[veh.detection_mask.astype(bool)] - 
				rewards_cleaning = self.n_trash_cleaned - (1 - np.array([
							np.sum(filtered_map[veh.detection_mask.astype(bool)]
									/ (np.sum(veh.detection_mask) * self.fleet.redundancy_mask[veh.detection_mask.astype(bool)]))
								for i,veh in enumerate(self.fleet.vehicles)]))
			
			if self.model_ant is not None:
				model_change = np.array([np.sum(np.abs(self.model[veh.detection_mask.astype(bool)] - self.model_ant[veh.detection_mask.astype(bool)])) for veh in self.fleet.vehicles])
				rewards_cleaning += model_change
		rewards = np.vstack((rewards_exploration, rewards_cleaning)).T
		#print(rewards)
		self.info = {}

		return {agent_id: rewards[agent_id] for agent_id in range(self.number_of_agents) if
				self.active_agents[agent_id]}

	def dijkstra(self, start):
		# Initialize distances and priority queue
		distances = {vertex: float('infinity') for vertex in self.graph}
		distances[start] = 0
		priority_queue = [(0, start)]
		predecessors = {vertex: None for vertex in self.graph}
		
		while priority_queue:
			current_distance, current_vertex = heapq.heappop(priority_queue)

			if current_distance > distances[current_vertex]:
				continue

			for neighbor, weight in self.graph[current_vertex].items():
				distance = current_distance + weight

				if distance < distances[neighbor]:
					distances[neighbor] = distance
					predecessors[neighbor] = current_vertex
					heapq.heappush(priority_queue, (distance, neighbor))
					
		return distances, predecessors

	def grid_to_graph(self,directions=None):
		rows = self.scenario_map.shape[0]
		cols = self.scenario_map.shape[1]
		graph = {}

		# Directions for 8 adjacent cells (including diagonals)
		if directions is None:
			directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

		for x in range(rows):
			for y in range(cols):
				if self.scenario_map[x,y] == 1:  # Assuming 1 represents a navigable cell
					graph[(x, y)] = {}
					for dx, dy in directions:
						nx, ny = x + dx, y + dy
						if 0 <= nx < rows and 0 <= ny < cols and self.scenario_map[nx,ny] == 1:
							if self.isnot_reachable(self.scenario_map, (x, y), (nx, ny)):
								continue
							graph[(x, y)][(nx, ny)] = np.linalg.norm(np.array([x,y]) - np.array([nx,ny]))  # Assuming all edges have a weight of 1

		return graph

	def calculate_distance_and_predecessor_maps(self):
		""" Calculate the distance and predecessor maps for each pixel """
		distance_map = {}
		predecessor_map = {}
		
		for pos in list(self.graph.keys()):
			i,j = pos
			distances, predecessors = self.dijkstra((i, j))
			distance_map[(i, j)] = distances
			predecessor_map[(i, j)] = predecessors
				
		return distance_map, predecessor_map

	def proportional_distance_value(self, map_values, position, alpha=1.0):

		distances = self.distance_map[position]
		penalization = 0.0
		# Calculate penalization
		for pos in np.argwhere(map_values > 0.):
			i,j = pos
			if distances[(i, j)] > 0:  # Penalize interesting pixels
				penalization += alpha * (map_values[i][j] / distances[(i, j)])
		
		return penalization

	def calculate_field_map(self,map_values, mask, alpha=1.0):
     
		field_map = np.zeros_like(mask, dtype=float)

		# Calculate the penalization for pixels where mask is True
		for pos in np.argwhere(mask):
			field_map[tuple(pos)] = self.proportional_distance_value(map_values, tuple(pos), alpha)

		return field_map


	def get_action_mask(self, ind=0):
		""" Return an array of Bools (True means this action for the agent ind causes a collision) """

		assert 0 <= ind < self.number_of_agents, 'Not enough agents!'

		return np.array(list(map(self.fleet.vehicles[ind].check_action, np.arange(0, 8))))
	

	def save_environment_configuration(self, path):
		""" Save the environment configuration in the current directory as a json file"""

		environment_configuration = {

			'number_of_agents': self.number_of_agents,
			'miopic': self.miopic,
			'fleet_initial_positions': self.initial_positions.tolist(),
			'distance_budget': self.distance_budget,
			'detection_length': self.detection_length,
			'movement_length': self.movement_length,
			'min_movements_if_nocollisions': self.min_movements_if_nocollisions,
			'max_number_of_colissions': self.max_collisions,
			'reward_type': self.reward_type,
			'ground_truth': self.ground_truth_type,
			'frame_stacking': self.num_of_frame_stacking,
			'state_index_stacking': self.state_index_stacking,
			'trail_length': self.trail_length

		}

		with open(path + '/environment_config.json', 'w') as f:
			json.dump(environment_configuration, f, indent=4)

	@staticmethod
	def isnot_reachable(grid, current_position, next_position):
		""" Check if the next position is reachable or navigable """
		if grid[int(next_position[0]), int(next_position[1])] == 0:
			return True 
		x, y = next_position
		dx = x - current_position[0]
		dy = y - current_position[1]
		steps = max(abs(dx), abs(dy))
		dx = dx / steps if steps != 0 else 0
		dy = dy / steps if steps != 0 else 0
		reachable_positions = True
		for step in range(1, steps + 1):
			px = round(current_position[0] + dx * step)
			py = round(current_position[1] + dy * step)
			if grid[px, py] != 1:
				reachable_positions = False
				break

		return not reachable_positions

if __name__ == '__main__':


	#sc_map = np.genfromtxt('Environment/Maps/example_map.csv', delimiter=',')
	sc_map = np.genfromtxt('Environment/Maps/malaga_port.csv', delimiter=',')

	N = 4
	initial_positions = np.array([[12, 7], [14, 5], [16, 3], [18, 1]])[:N, :]
	visitable = np.column_stack(np.where(sc_map == 1))
	initial_positions = visitable[np.random.randint(0,len(visitable), size=N), :]
	gts0 = []
	#initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])

	from tqdm import trange
	for _ in range(3):
		env = MultiAgentPatrolling(scenario_map=sc_map,
								fleet_initial_positions=initial_positions,
								distance_budget=200,
								number_of_vehicles=N,
								seed=43,
								miopic=True,
								detection_length=1,
								movement_length=1,
								max_collisions=500,
								ground_truth_type='macro_plastic',
								obstacles=False,
								frame_stacking=2,
								state_index_stacking=(0,1,2),
								reward_type='Distance Field',
								convert_to_uint8=False,
								trail_length = 20
												)
		reads = [2,4,9]
		#lengths = [20,100,33]
		gts = []
		n_actions = 8
		for k in trange(10):
			env.reset()
			lengths = 0
			done = {i:False for i in range(4)}

			R = []
			action = {i: np.random.randint(0,n_actions) for i in range(N)}

			while not all(done.values()):
				#action = {i: np.random.randint(0,8) for i in range(N)}
				for idx, agent in enumerate(env.fleet.vehicles):
				
					agent_mask = np.array([agent.check_action(a) for a in range(n_actions)], dtype=int)

					if agent_mask[action[idx]]:
						action[idx] = np.random.choice(np.arange(n_actions), p=(1-agent_mask)/np.sum((1-agent_mask)))
				s, r, done, _ = env.step(action)
				#print(env.steps)
				env.render()
				R.append(list(r.values()))
				lengths += 1
				for ke in range(N):
					if env.n_trash_cleaned[ke] > 0:
						print(env.n_trash_cleaned[ke])
				if k in reads:
					if lengths in [20,100,33]:
						gts.append(env.gt.read())
				#print(r)
		gts0.append(gts)

	env.render()
	plt.show()

	plt.plot(np.cumsum(np.asarray(R),axis=0), '-o')
	plt.xlabel('Step')
	plt.ylabel('Individual Reward')
	plt.legend([f'Agent {i}' for i in range(N)])
	plt.grid()
	plt.show()

# to print with colorbar 
"""fig,ax=plt.subplots()
im = ax.imshow(env.im1.get_array(),cmap='rainbow_r',vmin=0,vmax=1.0)
plt.colorbar(im,ax=ax)"""