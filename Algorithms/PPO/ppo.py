from Algorithms.PPO.policies import ActorCritic
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import time


class MemoryBuffer:
	""" This class is used to store the experiences of the agents and to put the data in the right format to train the policy. """
	""" The memory buffer is shared among all agents. An agent generates a trajectory and stores it in the memory buffer. """
	
	def __init__(self, max_samples, n_agents, action_size, obs_size, max_steps) -> None:
		
		self.max_samples = max_samples # Maximum number of trajectories that can be stored in the memory buffer
		self.n_agents = n_agents
		self.action_size = action_size
		self.obs_size = obs_size
		self.max_steps = max_steps  # Maximum number of steps in a trajectory
				
		# Initialize the lists to store the experiences
		self.actions = np.zeros((self.max_samples, self.n_agents, self.max_steps), dtype=np.float32)
		self.states = np.zeros((self.max_samples, self.n_agents, self.max_steps, *self.obs_size), dtype=np.float32)
		self.logprobs = np.zeros((self.max_samples, self.n_agents, self.max_steps), dtype=np.float32)
		self.rewards = np.zeros((self.max_samples, self.n_agents, self.max_steps), dtype=np.float32)
		self.dones = np.zeros((self.max_samples, self.n_agents, self.max_steps), dtype=np.float32)
		self.state_values = np.zeros((self.max_samples, self.n_agents, self.max_steps), dtype=np.float32)
		self.rewards_to_go = np.zeros((self.max_samples, self.n_agents, self.max_steps), dtype=np.float32)
		self.gae = np.zeros((self.max_samples, self.n_agents, self.max_steps), dtype=np.float32)
		
	def clear_buffer(self):
		""" Clear the memory buffer. """
		
		self.actions[:] = 0.0
		self.states[:] = 0.0
		self.logprobs[:] = 0.0
		self.rewards[:] = 0.0
		self.dones[:] = False
		self.state_values[:] = 0.0
		self.rewards_to_go[:] = 0.0
		self.gae[:] = 0.0

	def insert_experience(self, agent_id, sample_num, t, action, state, logprob, reward, done, state_value):
		""" Insert the experiences into the memory buffer. Every element is stored in the corresponding list. \
			For example, the actions will have the shape (n_agents, max_steps, action_size). """
			
		self.actions[sample_num, agent_id, t] = action
		self.states[sample_num, agent_id, t] = state
		self.logprobs[sample_num, agent_id, t] = logprob
		self.rewards[sample_num, agent_id, t] = reward
		self.dones[sample_num, agent_id, t] = done
		self.state_values[sample_num, agent_id, t] = state_value
		
	def compute_rewards_to_go(self, gamma):
		""" This function computes the rewards to go for each agent and for each trajectory. """
		
		for agent_id in range(self.n_agents):
			for i in range(self.max_samples):
				rewards_to_go = np.zeros(self.max_steps)
				for t in reversed(range(self.max_steps)):
					if t == self.max_steps - 1:
						rewards_to_go[t] = self.rewards[i, agent_id, t]
					else:
						rewards_to_go[t] = self.rewards[i, agent_id, t] + gamma * rewards_to_go[t+1] * (1 - self.dones[i, agent_id, t])
      
				self.rewards_to_go[i, agent_id, :] = (rewards_to_go - np.mean(rewards_to_go)) / (np.std(rewards_to_go) + 1e-5)
				#self.rewards_to_go[i, agent_id, :] = rewards_to_go
	
		
		
		
	def compute_gae(self, gamma, lamda):
		""" This function computes the Generalized Advantage Estimation for each agent and for each trajectory. """
		
		for agent_id in range(self.n_agents):
			for i in range(self.max_samples):
				gae = np.zeros(self.max_steps)
				
				for t in reversed(range(self.max_steps)):
					
					if t == self.max_steps - 1:
						td_error = self.rewards[i, agent_id, t] - self.state_values[i, agent_id, t]
						gae[t] = td_error
					else:
						td_error = self.rewards[i, agent_id, t] + gamma * self.state_values[i, agent_id, t+1] * (1 - self.dones[i, agent_id, t]) - self.state_values[i, agent_id, t]
						gae[t] = gae[t+1] * gamma * lamda * (1 - self.dones[i, agent_id, t]) + td_error

				# Normalize the GAE
				self.gae[i, agent_id, :] = (gae - np.mean(gae)) / (np.std(gae) + 1e-5)
				# self.gae[i, agent_id, :] = gae

			
	def get_batch(self, mini_batch_size):
		""" Get the trayectories batch from the memory buffer. """
  

		indices_traj = np.arange(self.max_samples)
		indices_step = np.arange(self.max_steps)
		agent_num_index = np.arange(self.n_agents)
  
		states = self.states.reshape(-1, *self.obs_size)
		actions = self.actions.reshape(-1)
		old_logprobs = self.logprobs.reshape(-1)
		state_values = self.state_values.reshape(-1)
		rewards_to_go = self.rewards_to_go.reshape(-1)
		gae = self.gae.reshape(-1)
  
		return states, actions, old_logprobs, state_values, rewards_to_go, gae

class PPO():
	'''Proximal Policy Optimization algorithm.'''
	def __init__(self, env, state_size, action_size, n_agents, max_steps, max_samples, eval_episodes, lr=3e-4, gamma=0.99, epsilon_clip=0.2, epochs=5, log = False, logdir = None, device = None):

		self.env = env  # Environment
		self.max_samples = max_samples  # Maximum number of trayectories that are sampled from the environment
		self.max_steps = max_steps # Max steps per trajectory
		self.state_size = state_size 
		self.n_agents = n_agents
		self.action_size = action_size
		self.lr = lr
		self.gamma  = gamma
		self.epsilon_clip = epsilon_clip
		self.K_epochs = epochs  # Number of epochs to train the policy
		self.num_of_minibatches = 32   # Number of minibatches to sample from the memory buffer
		self.log = log
		self.eval_episodes = eval_episodes

		# Automatic selection of the device 
		self.device = device
		torch.cuda.set_device(int(device[-1]))
		print("Selected device: ", self.device)

		# The policy is shared among all agents
		self.policy = ActorCritic(self.state_size, self.action_size, hidden_size=128).to(self.device)
		
		# The memory buffer is shared among all agents
		self.memory = MemoryBuffer(max_samples=max_samples,
									max_steps=max_steps,
							 		n_agents=self.n_agents,
							   		action_size=self.action_size,
								 	obs_size=self.state_size)
		
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
		
		if self.log:
			self.logdir = logdir
			self.writer = SummaryWriter(log_dir=self.logdir)
		

		
	@torch.no_grad()
	def act(self, state, greedy=False):
		""" Select an action from the policy """
		
		# state = torch.from_numpy(state).to(self.device)
		
		# Get the logits from the policy
		logits, values = self.policy(state)
		dist = torch.distributions.Categorical(logits=logits)
		if greedy:
			action = torch.argmax(logits, dim=-1)
		else:
			action = dist.sample()  # Sample an action from the policy
		
		return action, dist.log_prob(action), values
	
	
	
	def sample_experiences(self):
		""" Take a trajectory from the environment and store it in the memory buffer. """
		
		self.memory.clear_buffer()  # Clear the memory buffer

		self.mean_rewards_sample = []
		self.mean_cleaned_sample = []
		  
		for n_traj in range(self.max_samples):
      
			states = self.env.reset()
			states = torch.FloatTensor(states).to(self.device)
		
			for t in range(self.max_steps):
				# Get the actions and logprobs from the policy
				
				# Act for each agent # 
				
				actions, logprobs, state_values = self.act(states)
				
				# Take a step in the environment
				actions_np = actions.cpu().numpy()
				next_states, rewards, dones = self.env.step(actions_np)
				self.mean_rewards_sample.append(np.mean(rewards))
				self.mean_cleaned_sample.append(self.env.get_percentage_cleaned_trash())
				next_states = torch.FloatTensor(next_states).to(self.device)
				
				# Store the experiences in the memory buffer
				for agent_id in range(self.n_agents):
					self.memory.insert_experience(agent_id=agent_id, sample_num=n_traj, t=t, action=actions[agent_id].item(), state=states[agent_id].cpu().numpy(), logprob=logprobs[agent_id].item(), reward=rewards[agent_id].item(), done=dones[agent_id], state_value=state_values[agent_id].item())
				
				states = next_states
		self.mean_rewards_sample = np.mean(self.mean_rewards_sample)
		self.mean_cleaned_sample = np.mean(self.mean_cleaned_sample)
		print(f"Mean rewards in samples: {self.mean_rewards_sample}.")
		print(f"Mean cleaned in samples: {self.mean_cleaned_sample/100}%.")

								   
		
	def update_model(self, memory, iteration):
		""" Update the policy using the PPO algorithm. """
		
		# Compute the rewards to go and the Generalized Advantage Estimation
		memory.compute_rewards_to_go(self.gamma)
		memory.compute_gae(self.gamma, lamda=0.95)
  
		loss_policy = []
		loss_value = []
		loss_entropy = []
		max_estimated_values = []

		states_batch, actions_batch, old_logprobs_batch, state_values_batch, rewards_to_go_batch, gae_batch = memory.get_batch(self.max_samples)

		num_of_minibatches = 128
		minibatch_size = states_batch.shape[0] // num_of_minibatches

		for _ in range(self.K_epochs):
			
			loss = 0  # Total loss

			random_indices = np.random.permutation(states_batch.shape[0])

		
			for i in range(num_of_minibatches):

				# Sample the mini-batches randomly from the memory buffer
				indices = random_indices[i*minibatch_size:(i+1)*minibatch_size]

				states = states_batch[indices]
				actions = actions_batch[indices]
				old_logprobs = old_logprobs_batch[indices]
				state_values = state_values_batch[indices]
				rewards_to_go = rewards_to_go_batch[indices]
				gae = gae_batch[indices]
		
				# Compute the new logprobs and the ratio
				states = torch.FloatTensor(states).to(self.device)
				logits, state_values = self.policy(states)
				dist = torch.distributions.Categorical(logits=logits)
				actions = torch.LongTensor(actions).to(self.device).reshape(-1)
				new_logprobs = dist.log_prob(actions).reshape(-1)

				old_logprobs = torch.FloatTensor(old_logprobs).to(self.device).reshape(-1)
				ratio = torch.exp(new_logprobs - old_logprobs)
				
				# Compute the clipped surrogate loss
				clipped_ratio = torch.clamp(ratio, 1-self.epsilon_clip, 1+self.epsilon_clip)
				gae = torch.FloatTensor(gae).to(self.device).reshape(-1)
				clipped_surrogate = torch.min(ratio * gae.reshape(-1), clipped_ratio * gae).sum()
				
				# Compute the entropy loss
				entropy_loss = dist.entropy().sum()
				
				# Compute the clipped value loss
				rewards_to_go = torch.FloatTensor(rewards_to_go).to(self.device).reshape(-1)
				state_values = state_values.reshape(-1)
				value_loss = F.mse_loss(state_values, rewards_to_go, reduction="sum")

				max_estimated_values.append(torch.max(state_values).item())
				
				# Compute the total loss
				loss += -clipped_surrogate + 0.5 * value_loss - 0.01 * entropy_loss
			
			loss = loss / states_batch.shape[0] / num_of_minibatches

			# Optimize the policy
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			
			# Write the losses to tensorboard
			
			loss_policy.append(clipped_surrogate.mean().item())
			loss_value.append(value_loss.mean().item())
			loss_entropy.append(entropy_loss.mean().item())

		if self.log:
			self.writer.add_scalar("Loss/Clipped Surrogate", np.mean(loss_policy), iteration)
			self.writer.add_scalar("Loss/Value Loss", np.mean(loss_value), iteration)
			self.writer.add_scalar("Loss/Entropy Loss", np.mean(loss_entropy), iteration)
			self.writer.add_scalar("Value/Max estimated Value", np.max(max_estimated_values), iteration)
			self.writer.add_scalar("Mean reward sample", self.mean_rewards_sample, iteration)
			self.writer.add_scalar("Mean cleaned sample", self.mean_cleaned_sample, iteration)
				
		
	def train(self, n_iterations):
		""" Train the policy using the PPO algorithm. """
		
		BEST = -np.inf
		for iteration in trange(n_iterations):
			t0 = time.time()
			self.sample_experiences()
			print("Time to sample experiences: ", time.time() - t0)
			t0 = time.time()
			self.update_model(self.memory, iteration)
			print("Time to update model: ", time.time() - t0)
			rewards, cleaned_percentage = map(list, zip(*[self.evaluate_env() for _ in range(self.eval_episodes)]))
			MEAN = np.mean(rewards)
			mean_cleaned_percentage = np.mean(cleaned_percentage)
			if self.log:
				self.writer.add_scalar("Avg. Eval. Reward", MEAN, iteration)
				self.writer.add_scalar("Avg. Eval. CleanedP", mean_cleaned_percentage, iteration)
			if MEAN > BEST:
				BEST = MEAN
				if self.log:
					print(f"Saving new best model at iteration {iteration} with reward {MEAN}.")
					torch.save(self.policy.state_dict(), "best_model.pth")
		
		if self.log:
			torch.save(self.policy.state_dict(), "final_model.pth")
			self.writer.close()
	
	@torch.no_grad()
	def evaluate_env(self, render = False, greedy = False):
	 
		self.policy.eval()
	 
		states = self.env.reset()
		states = torch.FloatTensor(states).to(self.device)
		done = np.array([False]*self.n_agents)
  
		accumulated_reward = 0
		percentage_cleaned = 0
  
		while not np.any(done):
			action, _, _ = self.act(states, greedy=greedy)
			next_states, reward, done = self.env.step(action.cpu().numpy())
			states = torch.FloatTensor(next_states).to(self.device)
			accumulated_reward += np.sum(reward)
			if render:
				self.env.render()

		percentage_cleaned = self.env.get_percentage_cleaned_trash()
		self.policy.train()
  
		return accumulated_reward, percentage_cleaned
