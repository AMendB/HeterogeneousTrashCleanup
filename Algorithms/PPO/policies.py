import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class ActorCritic(nn.Module):
	def __init__(self, state_shape, action_size, hidden_size=32, low_policy_weights_init=True):
		super().__init__()
		""" Convolutional Neural Network for the actor and critic networks. """

		self.conv1 = nn.Conv2d(state_shape[0], 32, kernel_size=7, stride=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

		self._conv_out_size = self._get_conv_out(state_shape)

		self.actor_fc1 = nn.Linear(self._conv_out_size, 2*hidden_size)
		self.actor_fc2 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.actor_fc3 = nn.Linear(2*hidden_size, hidden_size)

		self.actor_output = nn.Linear(hidden_size, action_size)
		
		self.critic_fc1 = nn.Linear(self._conv_out_size, 2*hidden_size)
		self.critic_fc2 = nn.Linear(2*hidden_size, 2*hidden_size)
		self.critic_fc3 = nn.Linear(2*hidden_size, hidden_size)

		self.critic_value = nn.Linear(hidden_size, 1)

		# Categorical distribution for discrete actions
		self.distribution = torch.distributions.Categorical
		
	
	def _get_conv_out(self, shape):
		o = self.conv1(torch.zeros(1, *shape))
		o = self.conv2(o)
		o = self.conv3(o)
		return int(np.prod(o.size()))

	def forward(self, state):
		""" Forward pass for the actor and critic networks. """


		x = F.relu(self.conv1(state))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))

		# Flatten the output
		cnn_out = x.view(-1, self._conv_out_size)
		
		x = F.relu(self.actor_fc1(cnn_out))
		x = F.relu(self.actor_fc2(x))
		x = F.relu(self.actor_fc3(x))
  
		# Softmax for discrete actions
		logits = self.actor_output(x)
	
		v = F.relu(self.critic_fc1(cnn_out))
		v = F.relu(self.critic_fc2(v))
		v = F.relu(self.critic_fc3(v))
		state_value = self.critic_value(v)

		return logits, state_value 
