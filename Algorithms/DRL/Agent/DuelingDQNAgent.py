from typing import Dict, List, Tuple
from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Algorithms.DRL.ReplayBuffers.ReplayBuffers import PrioritizedReplayBuffer, ReplayBuffer
from Algorithms.DRL.Networks.network import DuelingVisualNetwork, NoisyDuelingVisualNetwork, DistributionalVisualNetwork
import torch.nn.functional as F
from tqdm import trange
from Algorithms.DRL.ActionMasking.ActionMaskingUtils import NoGoBackMasking, SafeActionMasking, ConsensusSafeActionMasking, NoGoBackFleetMasking
import time
import json
import os
from Algorithms.Greedy import OneStepGreedyFleet
from Algorithms.PSO import ParticleSwarmOptimizationFleet

class MultiAgentDuelingDQNAgent:

	def __init__(
			self,
			env: MultiAgentCleanupEnvironment, 
			memory_size: int,
			batch_size: int,
			target_update: int,
			soft_update: bool = False,
			tau: float = 0.0001,
			epsilon_values: List[float] = [1.0, 0.0],
			epsilon_interval: List[float] = [0.0, 1.0],
			greedy_training: bool = False,
			heuristic_training: bool = False,
			learning_starts: int = 10,
			gamma: float = 0.99,
			lr: float = 1e-4,
			logdir=None,
			log_name="Experiment",
			save_every=None,
			train_every=1,
			masked_actions= False,
			concensus_actions= False,
			device='cpu',
			seed = 0,
			eval_every = None,
			eval_episodes = 1000,
			prewarm_percentage = 0,
			# PER parameters
			alpha: float = 0.2,
			beta: float = 0.6,
			prior_eps: float = 1e-6,
			# NN parameters #
			number_of_features: int = 1024,
			noisy: bool = False,
			# Distributional parameters #
			distributional: bool = False,
			num_atoms: int = 51,
			v_interval: Tuple[float, float] = (0.0, 100.0),
			# A network for every team #
			independent_networks_per_team: bool = False,
			curriculum_learning_team = None,
	):
		"""

		:param env: Environment to optimize
		:param memory_size: Size of the experience replay
		:param batch_size: Mini-batch size for SGD steps
		:param target_update: Number of episodes between updates of the target
		:param soft_update: Flag to activate the Polyak update of the target
		:param tau: Polyak update constant
		:param gamma: Discount Factor
		:param lr: Learning Rate
		:param alpha: Randomness of the sample in the PER
		:param beta: Bias compensating constant in the PER weights
		:param prior_eps: Minimal probability for every experience to be samples
		:param number_of_features: Number of features after the visual extractor
		:param logdir: Directory to save the tensorboard log
		:param log_name: Name of the tb log
		"""

		""" Logging parameters """
		np.random.seed(seed)
		self.logdir = logdir
		self.experiment_name = log_name
		self.writer = None
		self.save_every = save_every
		self.eval_every = eval_every
		self.eval_episodes = eval_episodes

		""" Observation space dimensions """
		self.obs_dim = env.observation_space_shape
		self.action_dim_by_team = env.n_actions_by_team
		self.action_dim_of_each_agent = env.n_actions_of_each_agent

		""" Agent embeds the environment """
		self.env = env
		self.batch_size = batch_size
		self.target_update = target_update
		self.soft_update = soft_update
		self.tau = tau
		self.gamma = gamma
		self.learning_rate = lr
		self.epsilon_values = epsilon_values
		self.epsilon_interval = epsilon_interval
		self.epsilon = self.epsilon_values[0]
		self.greedy_training = greedy_training
		self.heuristic_training = heuristic_training
		self.learning_starts = learning_starts
		self.train_every = train_every
		self.masked_actions = masked_actions
		self.concensus_actions = concensus_actions
		self.noisy = noisy
		self.distributional = distributional
		self.num_atoms = num_atoms
		self.v_interval = v_interval
		self.independent_networks_per_team = independent_networks_per_team
		self.curriculum_learning_team = curriculum_learning_team

		""" Automatic selection of the device """
		self.device = device
		torch.cuda.set_device(int(device[-1]))

		print("Selected device: ", self.device)

		""" Prioritized Experience Replay """
		if self.independent_networks_per_team:
			self.memory = [PrioritizedReplayBuffer(self.obs_dim, memory_size, batch_size, alpha=alpha) if self.env.number_of_agents_by_team[team_id] > 0 else 0 for team_id in range(self.env.n_teams)]
		else:
			self.memory = PrioritizedReplayBuffer(self.obs_dim, memory_size, batch_size, alpha=alpha)
		self.beta = beta
		self.prior_eps = prior_eps
		self.prewarm_percentage = prewarm_percentage
		self.memory_size = memory_size

		""" Create the DQN and the DQN-Target (noisy if selected) """
		if self.independent_networks_per_team:
			self.dqn = [DuelingVisualNetwork(self.obs_dim, self.action_dim_by_team[team_id], number_of_features).to(self.device) if self.env.number_of_agents_by_team[team_id] > 0 else 0 for team_id in range(self.env.n_teams)]
			self.dqn_target = [DuelingVisualNetwork(self.obs_dim, self.action_dim_by_team[team_id], number_of_features).to(self.device) if self.env.number_of_agents_by_team[team_id] > 0 else 0 for team_id in range(self.env.n_teams)]
		elif self.noisy:
			self.dqn = NoisyDuelingVisualNetwork(self.obs_dim, action_dim, number_of_features).to(self.device)
			self.dqn_target = NoisyDuelingVisualNetwork(self.obs_dim, action_dim, number_of_features).to(self.device)
		elif self.distributional:
			self.support = torch.linspace(self.v_interval[0], self.v_interval[1], self.num_atoms).to(self.device)
			self.dqn = DistributionalVisualNetwork(self.obs_dim, action_dim, number_of_features, num_atoms, self.support).to(self.device)
			self.dqn_target = DistributionalVisualNetwork(self.obs_dim, action_dim, number_of_features, num_atoms, self.support).to(self.device)
		else:
			action_dim = self.action_dim_by_team[self.curriculum_learning_team]
			self.dqn = DuelingVisualNetwork(self.obs_dim, action_dim, number_of_features).to(self.device)
			self.dqn_target = DuelingVisualNetwork(self.obs_dim, action_dim, number_of_features).to(self.device)

		if self.independent_networks_per_team:
			# Load weights from dqn to dqn_target to be equal at the begining, and eval #
			for team_id in self.env.teams_ids:
				if self.env.number_of_agents_by_team[team_id] > 0:
					self.dqn_target[team_id].load_state_dict(self.dqn[team_id].state_dict())
					self.dqn_target[team_id].eval()
			""" Optimizers """
			self.optimizer = [optim.Adam(self.dqn[team_id].parameters(), lr=self.learning_rate) if self.env.number_of_agents_by_team[team_id] > 0 else 0 for team_id in self.env.teams_ids]
		else:
			# Load weights from dqn to dqn_target to be equal at the begining, and eval #
			self.dqn_target.load_state_dict(self.dqn.state_dict())
			self.dqn_target.eval()
			""" Optimizer """
			self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)

		# """ Actual list of transitions """
		# self.transition = list()

		""" Evaluation flag """
		# self.is_eval = False

		""" Data for logging """
		self.episodic_reward = []
		self.episodic_loss = []
		self.episodic_length = []
		self.episode = 0

		# Sample new noisy parameters
		if self.noisy:
			self.dqn.reset_noise()
			self.dqn_target.reset_noise()

		# Masking utilities #
		if self.concensus_actions:
			self.consensus_safe_masking_module = ConsensusSafeActionMasking(navigation_map = self.env.scenario_map, 
																   angle_set_of_each_agent= self.env.angle_set_of_each_agent, 
																   movement_length_of_each_agent = self.env.movement_length_of_each_agent)
			self.nogobackfleet_masking_module = NoGoBackFleetMasking()
		elif self.masked_actions:
			self.safe_masking_module = SafeActionMasking(action_space_dim = action_dim, movement_length = self.env.movement_length)
			self.nogoback_masking_modules = {i: NoGoBackMasking() for i in range(self.env.n_agents)}

	# TODO: Implement an annealed Learning Rate (see:
	#  https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)

	def predict_action(self, state: np.ndarray, deterministic: bool = False):

		"""Select an action from the input state. If deterministic, no noise is applied. """

		if self.epsilon > np.random.rand() and not self.noisy and not deterministic:
			selected_action = self.env.action_space.sample()

		else:
			q_values = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
			selected_action = np.argmax(q_values)

		return selected_action
	
	def select_actions(self, states: dict, deterministic: bool = False) -> dict:

		actions = {agent_id: self.predict_action(state, deterministic) for agent_id, state in states.items() }

		return actions

	def predict_masked_action(self, state: np.ndarray, agent_id: int, position: np.ndarray,  deterministic: bool = False):
		""" Select an action masked to avoid collisions and so """

		# Update the state of the safety module #
		self.safe_masking_module.update_state(position = position, new_navigation_map = state[0])

		if self.epsilon > np.random.rand() and not self.noisy and not deterministic:
			
			# Compute randomly the action #
			q_values, _ = self.safe_masking_module.mask_action(q_values = None)
			q_values, selected_action = self.nogoback_masking_modules[agent_id].mask_action(q_values = q_values)
			self.nogoback_masking_modules[agent_id].update_last_action(selected_action)

		else:
			q_values = self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
			q_values, _ = self.safe_masking_module.mask_action(q_values = q_values.flatten())
			q_values, selected_action = self.nogoback_masking_modules[agent_id].mask_action(q_values = q_values)
			self.nogoback_masking_modules[agent_id].update_last_action(selected_action)
		
		return selected_action

	def select_masked_actions(self, states: dict, positions: np.ndarray, deterministic: bool = False):

		actions = {agent_id: self.predict_masked_action(state=state, agent_id=agent_id, position=positions[agent_id], deterministic=deterministic) for agent_id, state in states.items()}

		return actions

	def select_concensus_actions(self, states: dict, positions: np.ndarray, n_actions_of_each_agent: int, done: dict, deterministic: bool = False):
		""" Select an action masked to avoid collisions and so """
		
		# Update navigation map if there are dynamic obstacles #
		# if self.env.osbtacles:
		# 	self.consensus_safe_masking_module.update_navigation_map(states[list(states.keys())[0]][0])

		if self.epsilon > np.random.rand() and not self.noisy and not deterministic:
			if self.greedy_training:
				if 0.5 > np.random.rand():
					# Greedy algorithm compute the q's #
					q_values = self.greedy_fleet.get_agents_q_values()
				else:
					# Compute randomly the q's #
					q_values = {agent_id: np.random.rand(n_actions_of_each_agent[agent_id]) for agent_id in states.keys() if not done[agent_id]}
			elif self.heuristic_training:
				rand_value = np.random.rand()
				if -1 > rand_value:
					# PSO algorithm compute the q's #
					q_values = self.pso_fleet.get_agents_q_values()
				elif 0.8 > rand_value:
					# Greedy algorithm compute the q's #
					q_values = self.greedy_fleet.get_agents_q_values()
				else:
					# Compute randomly the q's #
					q_values = {agent_id: np.random.rand(n_actions_of_each_agent[agent_id]) for agent_id in states.keys() if not done[agent_id]}
			else:
				# Compute randomly the q's #
				q_values = {agent_id: np.random.rand(n_actions_of_each_agent[agent_id]) for agent_id in states.keys() if not done[agent_id]}
		else:
			# The network compute the q's #
			if self.independent_networks_per_team:
				q_values = {agent_id: self.dqn[self.env.team_id_of_each_agent[agent_id]](torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy().flatten() for agent_id, state in states.items() if not done[agent_id]}
			else:
				q_values = {agent_id: self.dqn(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy().flatten() for agent_id, state in states.items() if not done[agent_id]}

		# Masking q's and take actions #
		q_values = self.nogobackfleet_masking_module.mask_actions(q_values=q_values)

		permanent_actions = self.consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=positions, model_trash_map=self.env.model_trash_map)
		self.nogobackfleet_masking_module.update_previous_actions(permanent_actions)
		
		return permanent_actions

	def step(self, action: dict) -> Tuple[np.ndarray, np.float64, bool]:
		"""Take an action and return the response of the env."""

		next_state, reward, done = self.env.step(action)

		return next_state, reward, done

	def update_model(self, team_id_index = None) -> torch.Tensor:
		"""Update the model by gradient descent."""

		# PER needs beta to calculate weights
		if self.independent_networks_per_team:
			samples = self.memory[team_id_index].sample_batch(self.beta)
			weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
			indices = samples["indices"]

			# PER: importance sampling before average
			elementwise_loss = self._compute_dqn_loss(samples, team_id_index)
			loss = torch.mean(elementwise_loss * weights)

			# Compute gradients and apply them
			self.optimizer[team_id_index].zero_grad()
			loss.backward()
			self.optimizer[team_id_index].step()

			# PER: update priorities
			loss_for_prior = elementwise_loss.detach().cpu().numpy()
			new_priorities = loss_for_prior + self.prior_eps
			self.memory[team_id_index].update_priorities(indices, new_priorities)

			# Sample new noisy distribution
			if self.noisy:
				self.dqn[team_id_index].reset_noise()
				self.dqn_target[team_id_index].reset_noise()

			return loss.item()
		
		else:
			samples = self.memory.sample_batch(self.beta)
			weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
			indices = samples["indices"]

			# PER: importance sampling before average
			elementwise_loss = self._compute_dqn_loss(samples)
			loss = torch.mean(elementwise_loss * weights)

			# Compute gradients and apply them
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			# PER: update priorities
			loss_for_prior = elementwise_loss.detach().cpu().numpy()
			new_priorities = loss_for_prior + self.prior_eps
			self.memory.update_priorities(indices, new_priorities)

			# Sample new noisy distribution
			if self.noisy:
				self.dqn.reset_noise()
				self.dqn_target.reset_noise()

			return loss.item()

	@staticmethod
	def anneal_epsilon(p, p_init=0.1, p_fin=0.9, e_init=1.0, e_fin=0.0):

		if p < p_init:
			return e_init
		elif p > p_fin:
			return e_fin
		else:
			return (e_fin - e_init) / (p_fin - p_init) * (p - p_init) + 1.0

	@staticmethod
	def anneal_beta(p, p_init=0.1, p_fin=0.9, b_init=0.4, b_end=1.0):

		if p < p_init:
			return b_init
		elif p > p_fin:
			return b_end
		else:
			return (b_end - b_init) / (p_fin - p_init) * (p - p_init) + b_init
	
	def prewarm_memory(self):
		"""Fill the memory with One Step Greedy experiences."""

		algorithm = OneStepGreedyFleet(env=self.env)

		print('Prewarming memory...')

		n_experiences_to_save = {team_id: self.memory_size*self.prewarm_percentage for team_id in self.env.teams_ids if self.env.number_of_agents_by_team[team_id] > 0}
		print(n_experiences_to_save)

		episode = 0
		while any(n_experiences_to_save.values()):

			episode += 1
			print(f'Prewarming episode {episode}...')
			
			done = {i: False for i in range(self.env.n_agents)}
			stop_saving = {i: not bool(n_experiences_to_save[self.env.team_id_of_each_agent[i]]) for i in range(self.env.n_agents)}
			states = self.env.reset_env()

			# Run an episode #
			while not all(done.values()):
				# Take new actions #
				actions = algorithm.get_agents_actions()

				# Process the agent step #
				next_states, reward, done = self.step(actions)

				# Store every observation for every agent #
				for agent_id in next_states.keys():
					if not stop_saving[agent_id]:
						transition = [states[agent_id],
											actions[agent_id],
											reward[agent_id],
											next_states[agent_id],
											done[agent_id],
											{}]
						if self.independent_networks_per_team:
							self.memory[self.env.team_id_of_each_agent[agent_id]].store(*transition)
						else:
							self.memory.store(*transition)
						stop_saving[agent_id] = done[agent_id]
						n_experiences_to_save[self.env.team_id_of_each_agent[agent_id]] = max(0, n_experiences_to_save[self.env.team_id_of_each_agent[agent_id]] - 1)

				# Update the state
				states = next_states

				# Break if the memory is full #
				if not any(n_experiences_to_save.values()):
					break

		print('Prewarming finished.')

	def train(self, episodes, extra_episodes=0):
		""" Train the agents. """

		self.episodes = episodes
		self.extra_episodes = extra_episodes

		# Prewarm memory #
		if self.prewarm_percentage > 0:
			self.prewarm_memory()

		# Use greedy policy to take actions for training instead of random #
		if self.greedy_training:
			self.greedy_fleet = OneStepGreedyFleet(env=self.env)
		elif self.heuristic_training:
			self.greedy_fleet = OneStepGreedyFleet(env=self.env)
			self.pso_fleet = ParticleSwarmOptimizationFleet(env=self.env)

		# START TRAINING #
		if self.independent_networks_per_team:
			
			# Percentage of experiences to store in memory #
			buffer_filled_percentage = 0.5 # percentage of training when the buffer is filled
			percentage_store_in_memory = {team_id: (self.memory_size * (1-self.prewarm_percentage))/((episodes*buffer_filled_percentage) * n_agents_in_team * self.env.max_steps_per_episode) for team_id, n_agents_in_team in enumerate(self.env.number_of_agents_by_team) if n_agents_in_team > 0}

			# Optimization steps per team #
			steps_per_team = [0]*self.env.n_teams
			
			# Create train logger and save configs #
			if self.writer is None:
				assert not os.path.exists(self.logdir), "El directorio ya existe. He evitado que se sobrescriba"
				if os.path.exists(self.logdir):
					self.logdir = self.logdir
				self.writer = [SummaryWriter(log_dir=self.logdir+f"/log{team_id}/", filename_suffix=f"_network{team_id}") if self.env.number_of_agents_by_team[team_id] > 0 else 0 for team_id in self.env.teams_ids]
				self.write_experiment_config()
				self.env.save_environment_configuration(self.logdir if self.logdir is not None else './')

			# Agent in training mode #
			# self.is_eval = False
			# Reset episode count #
			self.episode = [1]*self.env.n_teams
			# Reset metrics #
			episodic_reward_vector = [[]]*self.env.n_teams
			record = [-np.inf]*self.env.n_teams
			eval_record = [-np.inf]*self.env.n_teams
			eval_clean_record = [-np.inf]*self.env.n_teams
			eval_mse_record = [np.inf]*self.env.n_teams

			for episode in trange(1, int(episodes+extra_episodes) + 1):

				done = {i:False for i in range(self.env.n_agents)}
				states = self.env.reset_env()
				score = [0]*self.env.n_teams
				length = [0]*self.env.n_teams
				losses = [[]]*self.env.n_teams
				episode_finished_per_teams = {i:False if self.env.number_of_agents_by_team[i] > 0 else True for i in self.env.teams_ids}

				# Initially sample noisy policy #
				if self.noisy:
					for team_id in self.env.teams_ids:
						if self.env.number_of_agents_by_team[team_id] > 0:
							self.dqn[team_id].reset_noise()
							self.dqn_target[team_id].reset_noise()

				# PER: Increase beta temperature
				self.beta = self.anneal_beta(p=episode / episodes, p_init=0, p_fin=0.9, b_init=0.4, b_end=1.0)

				# Epsilon greedy annealing
				self.epsilon = self.anneal_epsilon(p=episode / episodes,
												p_init=self.epsilon_interval[0],
												p_fin=self.epsilon_interval[1],
												e_init=self.epsilon_values[0],
												e_fin=self.epsilon_values[1])

				# Run an episode #
				while not all(done.values()):

					# Increase the played steps per team #
					steps_per_team = [steps+1 if not episode_finished_per_teams[team_id] else steps for team_id, steps in enumerate(steps_per_team)]

					# Select the action using the current policy #
					actions = self.select_concensus_actions(states=states, positions=self.env.get_active_agents_positions_dict(), n_actions_of_each_agent=self.action_dim_of_each_agent, done = done)

					# Process the agent step #
					next_states, reward, done = self.step(actions)

					# Store every observation for every agent #
					for agent_id in next_states.keys():
						team_id = self.env.team_id_of_each_agent[agent_id]
						if True or np.random.rand() < percentage_store_in_memory[team_id] or self.memory[team_id].size == self.memory_size: # Store only a percentage of the experiences to fill the memory at the middle of the training
							transition = [states[agent_id],
												actions[agent_id],
												reward[agent_id],
												next_states[agent_id],
												done[agent_id],
												{}]

							self.memory[self.env.team_id_of_each_agent[agent_id]].store(*transition)

					# Update the state
					states = next_states

					reward_array = np.array([*reward.values()])

					for team_id in self.env.teams_ids:
						if self.env.number_of_agents_by_team[team_id] > 0 and not(episode_finished_per_teams[team_id]):
							# Accumulate indicators
							score[team_id] += np.mean(reward_array[self.env.masks_by_team[team_id]])  # The mean reward among the team
							length[team_id] += 1

						
							# If episode is ended for all agents of the team
							if self.env.dones_by_teams[team_id] == True:
								# Get info to save in tensorboard log #
								self.episodic_reward = score[team_id]
								self.episodic_length = length[team_id]
								self.episode[team_id] += 1

								# Append loss metric #
								if losses[team_id]:
									self.episodic_loss = np.mean(losses[team_id])

								# Compute average metrics #
								episodic_reward_vector[team_id].append(self.episodic_reward)

								# Log progress
								self.log_data(team_id_index=team_id)

								# Save policy if is better on average
								mean_episodic_reward = np.mean(episodic_reward_vector[team_id])
								if mean_episodic_reward > record[team_id]:
									print(f"\nNew best policy with mean reward of {mean_episodic_reward} [EP {episode}] for network nº {team_id}")
									print("Saving model in " + self.logdir)
									record[team_id] = mean_episodic_reward
									self.save_model(name=f'BestPolicy_network{team_id}.pth', team_id_index=team_id)
								
								# Set the episode ended for all agents of that team #
								episode_finished_per_teams[team_id] = True
			
							# If training is ready
							if len(self.memory[team_id]) >= self.batch_size and episode >= self.learning_starts:

								# Update model parameters by backprop-bootstrapping #
								if steps_per_team[team_id] % self.train_every == 0:

									loss = self.update_model(team_id_index=team_id)
									# Append loss #
									losses[team_id].append(loss)

								# Update target soft/hard #
								if self.soft_update:
									self._target_soft_update(team_id_index=team_id)
								elif episode % self.target_update == 0 and episode_finished_per_teams[team_id]:
									self._target_hard_update(team_id_index=team_id)

				# Save percentage of cleaned trash during the episode #
				if self.env.number_of_agents_by_team[self.env.cleaners_team_id] > 0:
					self.writer[self.env.cleaners_team_id].add_scalar('train/cleaned_percentage', self.env.get_percentage_cleaned_trash(), episode)
				
				# Save MSE error during the episode #
				if self.env.number_of_agents_by_team[self.env.explorers_team_id] > 0:
					self.writer[self.env.explorers_team_id].add_scalar('train/MSE_error', self.env.get_model_mse(), episode)

				# Save the model every N episodes #
				if self.save_every is not None:
					if episode % self.save_every == 0:
						for team_id in self.env.teams_ids:
							if self.env.number_of_agents_by_team[team_id] > 0:
								self.save_model(name=f'Episode_{episode}_Policy_network{team_id}.pth', team_id_index=team_id)

				# Reset previous actions of NoGoBack #
				self.nogobackfleet_masking_module.reset()

				# Reset PSO if heuristic training #
				if self.heuristic_training:
					self.pso_fleet.reset()

				# Evaluation #
				if self.eval_every is not None and episode % self.eval_every == 0:
					mean_eval_reward, mean_eval_length, mean_cleaned_percentage, mean_mse = self.evaluate_env(self.eval_episodes)
					for team_id in self.env.teams_ids:
						if self.env.number_of_agents_by_team[team_id] > 0:
							self.writer[team_id].add_scalar('test/accumulated_reward', mean_eval_reward[team_id], self.episode[team_id])
							self.writer[team_id].add_scalar('test/accumulated_length', mean_eval_length[team_id], self.episode[team_id])
							if team_id == self.env.cleaners_team_id:
								self.writer[team_id].add_scalar('test/mean_cleaned_percentage', mean_cleaned_percentage, self.episode[team_id])
							if team_id == self.env.explorers_team_id:
								self.writer[team_id].add_scalar('test/mean_mse', mean_mse, self.episode[team_id])
							if mean_eval_reward[team_id] > eval_record[team_id]:
								print(f"\nNew best policy (reward) IN EVAL with mean reward of {mean_eval_reward[team_id]} and cleaned percentage of {round(mean_cleaned_percentage*100,2)}% for network nº {team_id}")
								print("Saving model in " + self.logdir)
								eval_record[team_id] = mean_eval_reward[team_id]
								self.save_model(name=f'BestEvalPolicy_network{team_id}.pth', team_id_index=team_id)
							if mean_cleaned_percentage > eval_clean_record[team_id]:
								print(f"\nNew best policy (cleaned percentage) IN EVAL with mean reward of {mean_eval_reward[team_id]} and cleaned percentage of {round(mean_cleaned_percentage*100,2)}% for network nº {team_id}")
								print("Saving model in " + self.logdir)
								eval_clean_record[team_id] = mean_cleaned_percentage
								self.save_model(name=f'BestEvalCleanPolicy_network{team_id}.pth', team_id_index=team_id)
							if mean_mse < eval_mse_record[team_id]:
								print(f"\nNew best policy (MSE) IN EVAL with mean reward of {mean_eval_reward[team_id]} and cleaned percentage of {round(mean_cleaned_percentage*100,2)}% for network nº {team_id}")
								print("Saving model in " + self.logdir)
								eval_mse_record[team_id] = mean_mse
								self.save_model(name=f'BestEvalMSEPolicy_network{team_id}.pth', team_id_index=team_id)

			# Save the final policys #
			for team_id in self.env.teams_ids:
				if self.env.number_of_agents_by_team[team_id] > 0:
					self.save_model(name=f'Final_Policy_network{team_id}.pth', team_id_index=team_id)

		else:
			# Percentage of experiences to store in memory #
			buffer_filled_percentage = 0.5 # percentage of training when the buffer is filled
			percentage_store_in_memory = (self.memory_size * (1-self.prewarm_percentage))/((episodes*buffer_filled_percentage) * self.env.n_agents * self.env.max_steps_per_episode)

			# Optimization steps #
			steps = 0
			
			# Create train logger and save configs #
			if self.writer is None:
				assert not os.path.exists(self.logdir), "El directorio ya existe. He evitado que se sobrescriba"
				if os.path.exists(self.logdir):
					self.logdir = self.logdir
				self.writer = SummaryWriter(log_dir=self.logdir, filename_suffix=self.experiment_name)
				self.write_experiment_config()
				self.env.save_environment_configuration(self.logdir if self.logdir is not None else './')

			# Agent in training mode #
			# self.is_eval = False
			# Reset episode count #
			self.episode = 1
			# Reset metrics #
			episodic_reward_vector = []
			record = -np.inf
			eval_record = -np.inf

			for episode in trange(1, int(episodes) + 1):

				done = {i:False for i in range(self.env.n_agents)}
				states = self.env.reset_env()
				score = 0
				length = 0
				losses = []

				# Initially sample noisy policy #
				if self.noisy:
					self.dqn.reset_noise()
					self.dqn_target.reset_noise()

				# PER: Increase beta temperature
				self.beta = self.anneal_beta(p=episode / episodes, p_init=0, p_fin=0.9, b_init=0.4, b_end=1.0)

				# Epsilon greedy annealing
				self.epsilon = self.anneal_epsilon(p=episode / episodes,
												p_init=self.epsilon_interval[0],
												p_fin=self.epsilon_interval[1],
												e_init=self.epsilon_values[0],
												e_fin=self.epsilon_values[1])

				# Run an episode #
				while not all(done.values()):

					# Increase the played steps #
					steps += 1

					# Select the action using the current policy #
					if self.concensus_actions:
						actions = self.select_concensus_actions(states=states, positions=self.env.get_active_agents_positions_dict(), n_actions_of_each_agent=self.action_dim_of_each_agent, done = done)
					elif self.masked_actions:
						actions = self.select_masked_actions(states=states, positions=self.env.fleet.get_positions())
					else:
						actions = self.select_actions(states)
						actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]} # only active agents

					# Process the agent step #
					next_states, reward, done = self.step(actions)

					# Store every observation for every agent #
					for agent_id in next_states.keys():
						if np.random.rand() < percentage_store_in_memory or self.memory.size == self.memory_size: # Store only a percentage of the experiences to fill the memory at the middle of the training
							transition = [states[agent_id],
												actions[agent_id],
												reward[agent_id],
												next_states[agent_id],
												done[agent_id],
												{}]

							self.memory.store(*transition)

					# Update the state
					states = next_states
					# Accumulate indicators
					score += np.mean(list(reward.values()))  # The mean reward among the agents
					length += 1

					# if episode ends
					if all(done.values()):

						# Append loss metric #
						if losses:
							self.episodic_loss = np.mean(losses)

						# Compute average metrics #
						self.episodic_reward = score
						self.episodic_length = length
						episodic_reward_vector.append(self.episodic_reward)
						self.episode += 1

						# Log progress
						self.log_data()

						# Save policy if is better on average
						mean_episodic_reward = np.mean(episodic_reward_vector)
						if mean_episodic_reward > record:
							print(f"\nNew best policy with mean reward of {mean_episodic_reward}")
							print("Saving model in " + self.writer.log_dir)
							record = mean_episodic_reward
							self.save_model(name='BestPolicy.pth')

					# If training is ready
					if len(self.memory) >= self.batch_size and episode >= self.learning_starts:

						# Update model parameters by backprop-bootstrapping #
						if steps % self.train_every == 0:

							loss = self.update_model()
							# Append loss #
							losses.append(loss)

						# Update target soft/hard #
						if self.soft_update:
							self._target_soft_update()
						elif episode % self.target_update == 0 and all(done.values()):
							self._target_hard_update()

				if self.save_every is not None:
					if episode % self.save_every == 0:
						self.save_model(name=f'Episode_{episode}_Policy.pth')

				# Reset previous actions of NoGoBack #
				self.nogobackfleet_masking_module.reset()

				# Evaluation #
				if self.eval_every is not None:
					if episode % self.eval_every == 0:
						mean_eval_reward, mean_eval_length, mean_cleaned_percentage, mean_mse = self.evaluate_env(self.eval_episodes)
						self.writer.add_scalar('test/accumulated_reward', mean_eval_reward, self.episode)
						self.writer.add_scalar('test/accumulated_length', mean_eval_length, self.episode)
						self.writer.add_scalar('test/mean_cleaned_percentage', mean_cleaned_percentage, self.episode)
						self.writer.add_scalar('test/mean_mse', mean_mse, self.episode)
						if mean_eval_reward > eval_record:
								print(f"\nNew best policy IN EVAL with mean reward of {mean_eval_reward}")
								print("Saving model in " + self.logdir)
								eval_record = mean_eval_reward
								self.save_model(name='BestEvalPolicy.pth')	

			# Save the final policy #
			self.save_model(name='Final_Policy.pth')

	def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], team_id_index = None) -> torch.Tensor:

		"""Return dqn loss."""
		device = self.device  # for shortening the following lines
		states = torch.FloatTensor(samples["obs"]).to(device)
		next_states = torch.FloatTensor(samples["next_obs"]).to(device)
		actions = torch.LongTensor(samples["acts"]).to(device)
		rewards = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
		dones = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

		# G_t   = r + gamma * v(s_{t+1})  if state != Terminal
		#       = r                       otherwise

		if self.independent_networks_per_team:
			actions = actions.reshape(-1, 1)
			curr_q_value = self.dqn[team_id_index](states).gather(1, actions)
			dones_mask = 1 - dones

			with torch.no_grad():
				next_q_value = self.dqn_target[team_id_index](next_states).gather(1, self.dqn[team_id_index](next_states).argmax(dim=1, keepdim=True))
				target = (rewards + self.gamma * next_q_value * dones_mask).to(self.device)

			# calculate element-wise dqn loss
			elementwise_loss = F.mse_loss(curr_q_value, target, reduction="none")

		elif self.distributional:
			# Distributional Q-Learning - Here is where the fun begins #
			delta_z = float(self.v_interval[1] - self.v_interval[0]) / (self.num_atoms - 1)

			with torch.no_grad():

				# max_a = argmax_a' Q'(s',a')
				next_action = self.dqn_target(next_states).argmax(1)
				# V'(s', max_a)
				next_dist = self.dqn_target.dist(next_states)
				next_dist = next_dist[range(self.batch_size), next_action]

				# Compute the target distribution by adding the
				t_z = rewards + (1 - dones) * self.gamma * self.support
				t_z = t_z.clamp(min=self.v_interval[0], max=self.v_interval[1])
				b = (t_z - self.v_interval[0]) / delta_z
				lower_bound = b.floor().long()
				upper_bound = b.ceil().long()

				offset = (torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size
					).long()
					.unsqueeze(1)
					.expand(self.batch_size, self.num_atoms)
					.to(self.device)
				)

				proj_dist = torch.zeros(next_dist.size(), device=self.device)
				proj_dist.view(-1).index_add_(
					0, (lower_bound + offset).view(-1), (next_dist * (upper_bound.float() - b)).view(-1)
				)
				proj_dist.view(-1).index_add_(
					0, (upper_bound + offset).view(-1), (next_dist * (b - lower_bound.float())).view(-1)
				)

			dist = self.dqn.dist(states)
			log_p = torch.log(dist[range(self.batch_size), actions])

			elementwise_loss = -(proj_dist * log_p).sum(1)

		else:
			actions = actions.reshape(-1, 1)
			curr_q_value = self.dqn(states).gather(1, actions)
			dones_mask = 1 - dones

			with torch.no_grad():
				next_q_value = self.dqn_target(next_states).gather(1, self.dqn(next_states).argmax(dim=1, keepdim=True))
				target = (rewards + self.gamma * next_q_value * dones_mask).to(self.device)

			# calculate element-wise dqn loss
			elementwise_loss = F.mse_loss(curr_q_value, target, reduction="none")

		return elementwise_loss

	def _target_hard_update(self, team_id_index=None):
		"""Hard update: target <- local."""

		if self.independent_networks_per_team:
			print(f"Hard update performed at episode {self.episode[team_id_index]} for network {team_id_index}!")
			self.dqn_target[team_id_index].load_state_dict(self.dqn[team_id_index].state_dict())
		else:
			print(f"Hard update performed at episode {self.episode}!")
			self.dqn_target.load_state_dict(self.dqn.state_dict())

	def _target_soft_update(self, team_id_index=None):
		"""Soft update: target_{t+1} <- local * tau + target_{t} * (1-tau)."""

		if self.independent_networks_per_team:
			for target_param, local_param in zip(self.dqn_target[team_id_index].parameters(), self.dqn_target[team_id_index].parameters()):
				target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
		else:
			for target_param, local_param in zip(self.dqn_target.parameters(), self.dqn_target.parameters()):
				target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

	def log_data(self, team_id_index=None):
		""" Logs for tensorboard """

		if self.independent_networks_per_team:
			if self.episodic_loss:
				self.writer[team_id_index].add_scalar('train/loss', self.episodic_loss, self.episode[team_id_index])

			self.writer[team_id_index].add_scalar('train/epsilon', self.epsilon, self.episode[team_id_index])
			self.writer[team_id_index].add_scalar('train/beta', self.beta, self.episode[team_id_index])

			self.writer[team_id_index].add_scalar('train/accumulated_reward', self.episodic_reward, self.episode[team_id_index])
			self.writer[team_id_index].add_scalar('train/accumulated_length', self.episodic_length, self.episode[team_id_index])

			self.writer[team_id_index].add_scalar('train/buffer_size', self.memory[team_id_index].size, self.episode[team_id_index])

			self.writer[team_id_index].flush()

		else:
			if self.episodic_loss:
				self.writer.add_scalar('train/loss', self.episodic_loss, self.episode)

			self.writer.add_scalar('train/epsilon', self.epsilon, self.episode)
			self.writer.add_scalar('train/beta', self.beta, self.episode)

			self.writer.add_scalar('train/accumulated_reward', self.episodic_reward, self.episode)
			self.writer.add_scalar('train/accumulated_length', self.episodic_length, self.episode)

			self.writer.add_scalar('train/buffer_size', self.memory.size, self.episode)

			self.writer.flush()

	def load_model(self, path_to_file):
		
		if self.independent_networks_per_team:
			for team_id in self.env.teams_ids:
				if self.env.number_of_agents_by_team[team_id] > 0:
					self.dqn[team_id].load_state_dict(torch.load(path_to_file[:-4] + f'_network{team_id}.pth', map_location=self.device))
					# Load weights from dqn to dqn_target to be equal at the begining #
					self.dqn_target[team_id].load_state_dict(self.dqn[team_id].state_dict())
		else:
			self.dqn.load_state_dict(torch.load(path_to_file, map_location=self.device))
			# Load weights from dqn to dqn_target to be equal at the begining #
			self.dqn_target.load_state_dict(self.dqn.state_dict())

	def save_model(self, name='experiment.pth', team_id_index=None):
		
		if self.independent_networks_per_team:
			try:
				torch.save(self.dqn[team_id_index].state_dict(), self.logdir + '/' + name)
			except:
				print(f"Model {name} saving failed!")
				try:
					time.sleep(3)
					torch.save(self.dqn[team_id_index].state_dict(), self.logdir + '/' + name)
					print("Done!")
				except:
					print('Definitely not saved :(')
		else:
			try:
				torch.save(self.dqn.state_dict(), self.writer.log_dir + '/' + name)
			except:
				print("Model saving failed!")
				try:
					time.sleep(3)
					torch.save(self.dqn.state_dict(), self.writer.log_dir + '/' + name)
					print("Done!")
				except:
					print('Definitely not saved :(')

	def evaluate_env(self, eval_episodes):
		""" Evaluate the agent on the environment for a given number of episodes with a deterministic policy """
		if self.independent_networks_per_team:
			total_reward = np.array([0]*self.env.n_teams)
			total_length = np.array([0]*self.env.n_teams)
			cleaned_percentage = []
			mse_error_accumulated = 0
			
			# Set networks to eval #
			for team_id in self.env.teams_ids:
				if self.env.number_of_agents_by_team[team_id] > 0:
					self.dqn[team_id].eval()

			for _ in trange(eval_episodes):

				# Reset the environment #
				states = self.env.reset_env()
				done = {agent_id: False for agent_id in range(self.env.n_agents)}
				# acc_r = np.array([0]*self.env.n_agents)
				

				while not all(done.values()):

					states = {agent_id: np.float16(np.uint8(state * 255)/255) for agent_id, state in states.items()}

					# Select the action using the current policy
					if self.concensus_actions:
						actions = self.select_concensus_actions(states=states, positions=self.env.get_active_agents_positions_dict(), n_actions_of_each_agent=self.action_dim_of_each_agent, done = done, deterministic=True)
					elif self.masked_actions:
						actions = self.select_masked_actions(states=states, positions=self.env.fleet.get_positions())
					else:
						actions = self.select_actions(states)
						actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]} # only active agents

					# Process the agent step #
					states, reward, done = self.step(actions)
					# print(f"Reward: {reward}")

					reward_array = np.array([*reward.values()])
					mse_error_accumulated += self.env.get_model_mse()

					for team_id in self.env.teams_ids:
						if self.env.number_of_agents_by_team[team_id] > 0:
							if not(self.env.dones_by_teams[team_id]):
								total_length[team_id] += 1
								total_reward[team_id] += np.mean(reward_array[self.env.masks_by_team[team_id]])
					# acc_r = acc_r + reward_array
				# print(f"Accumulated reward (EP: {_}: {acc_r}")
				
				# Save percentage of cleaning #
				cleaned_percentage.append(self.env.get_percentage_cleaned_trash())

				# Reset previous actions of NoGoBack #
				self.nogobackfleet_masking_module.reset()

			# Set networks to train #
			for team_id in self.env.teams_ids:
				if self.env.number_of_agents_by_team[team_id] > 0:
					self.dqn[team_id].train()
					print(f'\n Average eval reward team {team_id}: {total_reward[team_id]/eval_episodes}. Episode average length: {total_length[team_id] / eval_episodes}. Mean cleaned: {round(np.mean(cleaned_percentage)*100,2)}%. Mean MSE accumulated: {round(mse_error_accumulated / eval_episodes, 4)}')

		else:
			self.dqn.eval()
			total_reward = 0
			total_length = 0
			cleaned_percentage = []
			mse_error_accumulated = 0

			for _ in trange(eval_episodes):

				# Reset the environment #
				states = self.env.reset_env()
				done = {agent_id: False for agent_id in range(self.env.n_agents)}
				

				while not all(done.values()):

					total_length += 1

					# Select the action using the current policy
					if self.concensus_actions:
						actions = self.select_concensus_actions(states=states, positions=self.env.get_active_agents_positions_dict(), n_actions_of_each_agent=self.action_dim_of_each_agent, done = done, deterministic=True)
					elif self.masked_actions:
						actions = self.select_masked_actions(states=states, positions=self.env.fleet.get_positions())
					else:
						actions = self.select_actions(states)
						actions = {agent_id: action for agent_id, action in actions.items() if not done[agent_id]} # only active agents

					# Process the agent step #
					states, reward, done = self.step(actions)
					
					total_reward += np.sum(list(reward.values()))
					mse_error_accumulated += self.env.get_model_mse()
				
				# Save percentage of cleaning #
				cleaned_percentage.append(self.env.get_percentage_cleaned_trash())

				# Reset previous actions of NoGoBack #
				self.nogobackfleet_masking_module.reset()

			self.dqn.train()

		# Return the average reward, average length
		return total_reward / eval_episodes, total_length / eval_episodes, np.mean(cleaned_percentage), mse_error_accumulated / eval_episodes

	def write_experiment_config(self):
		""" Write experiment and environment variables in a json file """

		self.experiment_config = {
			"save_every": self.save_every,
			"eval_every": self.eval_every,
			"eval_episodes": self.eval_episodes,
			"batch_size": self.batch_size,
			"gamma": self.gamma,
			"tau": self.tau,
			"lr": self.learning_rate,
			"epsilon": self.epsilon,
			"epsilon_values": self.epsilon_values,
			"epsilon_interval": self.epsilon_interval,
			"train_every": self.train_every,
			"greedy_training": self.greedy_training,
			"heuristic_training": self.heuristic_training,
			"beta": self.beta,
			"num_atoms": self.num_atoms,
			"masked_actions": self.masked_actions,
			"concensus_actions": self.concensus_actions,
			"independent_networks_per_team": self.independent_networks_per_team,
			"curriculum_learning_team": self.curriculum_learning_team,
			"soft_update": self.soft_update,
			"target_update": self.target_update,
			"prewarm_percentage": self.prewarm_percentage,
			"episodes": self.episodes,
			"extra_episodes": self.extra_episodes,
			"memory_size": self.memory_size,
		}

		with open(self.logdir + '/experiment_config.json', 'w') as f:

			json.dump(self.experiment_config, f, indent=4)
