import sys
sys.path.append('.')

from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np

# Selection of PARAMETERS TO TRAIN #
reward_function = 'basic_reward' # basic_reward
reward_weights = (10, 10, 0) 
memory_size = int(1E4)
network_type = 'independent_networks_per_team'
device = 'cuda:0'
episodes = 60000
n_agents = 4  # max 4







SHOW_PLOT_GRAPHICS = False
seed = 0

# Agents info #
n_actions = 9
n_explorers = 2
n_cleaners = 2
n_agents = n_explorers + n_cleaners
movement_length_explorers = 2
movement_length_cleaners = 1
movement_length_of_each_agent = np.repeat((movement_length_explorers, movement_length_cleaners), (n_explorers, n_cleaners))
vision_length_explorers = 1
vision_length_cleaners = 4
max_distance_travelled_explorers = 400
max_distance_travelled_cleaners = 200


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
env = MultiAgentCleanupEnvironment(scenario_map = scenario_map,
							number_of_agents_by_team=(n_explorers,n_cleaners),
							n_actions=n_actions,
							max_distance_travelled_by_team = (max_distance_travelled_explorers, max_distance_travelled_cleaners),
							fleet_initial_positions = initial_positions, # None, 'area', 'fixed' or positions array
							seed = seed,
							movement_length_by_team =  (movement_length_explorers, movement_length_cleaners),
							vision_length_by_team = (vision_length_explorers, vision_length_cleaners),
							flag_to_check_collisions_within = True,
							max_collisions = 10,
							reward_function = reward_function, 
							reward_weights = reward_weights,
							dynamic = True,
							obstacles = False,
							show_plot_graphics = SHOW_PLOT_GRAPHICS,
							)

# Network config:
if network_type == 'independent_networks_per_team':
	independent_networks_per_team = True

if memory_size == int(1E3):
	logdir = f'testing/Training_{network_type.split("_")[0]}_RW_{reward_function.split("_")[0]}_' + '_'.join(map(str, reward_weights))
else:
	logdir = f'Training/Trning_RW_{reward_function.split("_")[0]}_' + '_'.join(map(str, reward_weights))

network = MultiAgentDuelingDQNAgent(env=env,
									memory_size=memory_size,  #int(1E6), 1E5
									batch_size=128,
									target_update=1000,
									soft_update=True,
									tau=0.001,
									epsilon_values=[1.0, 0.05],
									epsilon_interval=[0.0, 0.33], #0.5
									learning_starts=100, 
									gamma=0.99,
									lr=1e-4,
									save_every=5000, # 5000
									train_every=4, #15
									masked_actions=False,
									concensus_actions=True,
									device=device,
									logdir=logdir,
									eval_episodes=50, # 10
									eval_every=1000, #1000
									noisy=False,
									distributional=False,
									independent_networks_per_team = independent_networks_per_team,
)

network.train(episodes=episodes) 