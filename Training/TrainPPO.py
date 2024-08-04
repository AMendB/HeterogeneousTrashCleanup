import sys
sys.path.append('.')

from Environment.EnvPPOWrapper import EnvWrapper
from Algorithms.PPO.ppo import PPO
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-rw', '--reward_function', type=str, default='backtosimpledistanceppo', help='Reward function to use: basic_reward, extended_reward, backtosimple')
parser.add_argument('-w', '--reward_weights', type=int, nargs='+', default=[1, 25, 2, 10], help='Reward weights for the reward function.')
parser.add_argument('-net', '--network_type', type=str, default='independent_networks_per_team', help='Type of network to use: independent_networks_per_team, shared_network')
parser.add_argument('-dev', '--device', type=str, default='cuda:0', help='Device to use: cuda:x, cpu')
parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations to train the network.')
parser.add_argument('--extra_name', type=str, default='', help='Extra name to add to the logdir.')
parser.add_argument('--preload_path', type=str, default='', help='Path to preload a model.')
args = parser.parse_args()

# Selection of PARAMETERS TO TRAIN #
iterations = args.iterations
reward_function = args.reward_function
reward_weights = tuple(args.reward_weights) 
network_type = args.network_type
device = args.device
preload_path = args.preload_path







SHOW_PLOT_GRAPHICS = False
seed = 0

# Agents info #
n_actions_explorers = 8
n_actions_cleaners = 8
n_explorers = 0
n_cleaners = 1
n_agents = n_explorers + n_cleaners
movement_length_explorers = 2
movement_length_cleaners = 1
movement_length_of_each_agent = np.repeat((movement_length_explorers, movement_length_cleaners), (n_explorers, n_cleaners))
vision_length_explorers = 4
vision_length_cleaners = 1
max_distance_travelled_explorers = 400
max_distance_travelled_cleaners = 200
max_steps_per_episode = 150


# scenario_map = np.genfromtxt('Environment/Maps/ypacarai_map_low_res.csv', delimiter=',')
scenario_map = np.genfromtxt('Environment/Maps/acoruna_port.csv', delimiter=',')

# Set initial positions #
random_initial_positions = True 
if random_initial_positions:
	initial_positions = 'fixed'
else:
	# initial_positions = np.array([[46, 28], [46, 31], [49, 28], [49, 31]])[:n_agents, :] #ypacarai_map
	initial_positions = np.array([[32, 7], [30, 7], [28, 7], [26, 7]])[:n_agents, :] #coruna_port

# Create environment # 
env = EnvWrapper(scenario_map = scenario_map,
				number_of_agents_by_team=(n_explorers,n_cleaners),
				n_actions_by_team=(n_actions_explorers, n_actions_cleaners),
				max_distance_travelled_by_team = (max_distance_travelled_explorers, max_distance_travelled_cleaners),
				max_steps_per_episode = max_steps_per_episode,
				fleet_initial_positions = initial_positions, # None, 'area', 'fixed' or positions array
				seed = seed,
				movement_length_by_team =  (movement_length_explorers, movement_length_cleaners),
				vision_length_by_team = (vision_length_explorers, vision_length_cleaners),
				flag_to_check_collisions_within = False,
				max_collisions = 10,
				reward_function = reward_function, 
				reward_weights = reward_weights,
				dynamic = False,
				obstacles = False,
				show_plot_graphics = SHOW_PLOT_GRAPHICS,
				)

# Network config:
if network_type == 'independent_networks_per_team':
	independent_networks_per_team = True
else:
	independent_networks_per_team = False

if n_explorers == 0 or n_cleaners == 0:
	logdir = f'Training/T_PPO_curriculum_RW_{reward_function.split("_")[0]}_' + '_'.join(map(str, reward_weights)) + f'_{int(iterations/1000)}k' + args.extra_name
else:
	logdir = f'Training/T_PPO_RW_{reward_function.split("_")[0]}_' + '_'.join(map(str, reward_weights)) + f'_{int(iterations/1000)}k' + args.extra_name

state_size = env.observation_space_shape
action_size = n_actions_cleaners
n_agents = n_cleaners
max_steps = env.max_steps_per_episode
max_samples = 10
ppo = PPO(env=env, 
		  state_size=state_size, 
		  action_size=action_size, 
		  n_agents=n_agents, 
		  max_steps=max_steps, 
		  max_samples=max_samples, 
		  eval_episodes=50,
		  log=True, 
		  logdir=logdir, 
		  device=device)
if preload_path:
	ppo.load_model(preload_path)
ppo.train(n_iterations=iterations)
