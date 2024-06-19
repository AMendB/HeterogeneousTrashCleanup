import numpy as np

class SafeActionMasking:

	def __init__(self, action_space_dim: int, movement_length: float) -> None:
		""" Safe Action Masking """

		self.navigation_map = None
		self.position = None
		self.angle_set = np.linspace(0, 2 * np.pi, action_space_dim, endpoint=False)
		self.movement_length = movement_length

	def update_state(self, position: np.ndarray, new_navigation_map: np.ndarray = None):
		""" Update the navigation map """

		if new_navigation_map is not None:
			self.navigation_map = new_navigation_map

		""" Update the position """
		self.position = position

	def mask_action(self, q_values: np.ndarray = None):

		if q_values is None:
			""" Random selection """
			q_values = np.random.rand(8)

		movements = np.array([np.round(np.array([np.cos(angle), np.sin(angle)]) * self.movement_length ).astype(int) for angle in self.angle_set])
		next_positions = self.position + movements

		action_mask = np.array([self.navigation_map[int(next_position[0]), int(next_position[1])] == 0 for next_position in next_positions]).astype(bool)

		q_values[action_mask] = -np.inf

		return q_values, np.argmax(q_values)

class NoGoBackMasking:

	def __init__(self) -> None:
		
		self.previous_action = None

	def mask_action(self, q_values: np.ndarray = None):

		if q_values is None:
			""" Random selection """
			q_values = np.random.rand(8)

		if self.previous_action is None:
			self.previous_action = np.argmax(q_values)
		else:
			return_action = (self.previous_action + len(q_values) // 2) % len(q_values)
			q_values[return_action] = -np.inf

		return q_values, np.argmax(q_values)

	def update_last_action(self, last_action):

		self.previous_action = last_action
	
class NoGoBackFleetMasking:

	def __init__(self) -> None:

		self.reset()
		
	def reset(self):

		self.previous_actions = None

	def mask_actions(self, q_values: dict):

		if self.previous_actions is None:
			self.previous_actions = {idx: np.argmax(q_values[idx]) for idx in q_values.keys()}
		else:
			# Find the action that would make the agent go back
			return_actions = {idx: (self.previous_actions[idx] + len(q_values[idx]) // 2) % len(q_values[idx]) for idx in q_values.keys()}
			for idx in q_values.keys():
				if self.previous_actions[idx] < 8: # if the previous action induced a movement, i.e., it was not to stay in the same position
					q_values[idx][return_actions[idx]] = -1000 # a very low value instead of -np.inf to not have probability of collide with obstacle in random select in case of no alternative way out

		return q_values

	def update_previous_actions(self, previous_actions):

		self.previous_actions = previous_actions

class ConsensusSafeActionMasking:
	""" The optimists decide first! """

	def __init__(self, navigation_map, angle_set_of_each_agent: dict, movement_length_of_each_agent) -> None:
		
		self.navigation_map = navigation_map
		self.obstacles_map = self.navigation_map.copy()
		self.potential_movements_of_each_agent = {agent_id: np.array([(0,0) if angle < 0 else np.round([np.cos(angle), np.sin(angle)]) * movement_length_of_each_agent[agent_id] 
													 for angle in angle_set_of_each_agent[agent_id]]).astype(int) for agent_id in angle_set_of_each_agent.keys()}

	def update_map(self, new_navigation_map: np.ndarray):

		self.navigation_map = new_navigation_map.copy()

	def query_actions(self, q_values: dict, agents_positions: dict, ):

		# 1) The largest q-value agent decides first
		# 2) If there are multiple agents with the same q-value, the agent is selected randomly
		# 3) Then, compute the next position of the agent and update the fleet map
		# 4) The next agent is selected based on the updated fleet map, etc
		
		self.obstacles_map = self.navigation_map.copy()
		q_max = {idx: q_values[idx].max() for idx in q_values.keys()}
		agents_order = sorted(q_max, key=q_max.get)[::-1]
		final_actions = {}

		for agent_id in agents_order:
			
			# Compute all next possible positions
			next_positions = agents_positions[agent_id] + self.potential_movements_of_each_agent[agent_id]
			next_positions = np.clip(next_positions, (0,0), np.array(self.obstacles_map.shape)-1) # saturate movement if out of indexes values (map edges)
			
			# Check which next possible positions lead to a collision
			actions_mask = np.array([self.obstacles_map[int(next_position[0]), int(next_position[1])] == 0 for next_position in next_positions]).astype(bool)

			# Censor the impossible actions in the Q-values
			q_values[agent_id][actions_mask] = -np.inf

			# Select the action
			action = np.argmax(q_values[agent_id])

			# Update the obstacles map with the positions where the agent will move, so the next agents will consider them as obstacles
			next_position = next_positions[action]
			self.obstacles_map[int(next_position[0]), int(next_position[1])] = 0

			# Store the action
			final_actions[agent_id] = action.copy()


		return final_actions 
		
