import sys
import json
sys.path.append('.')

from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np

path_to_training_folder = 'Training/Trning_RW_basic_10_10_0/'

f = open(path_to_training_folder + 'environment_config.json',)
env_config = json.load(f)
f.close()

SEED = 3
SHOW_PLOT_GRAPHICS = True

env = MultiAgentCleanupEnvironment(scenario_map = np.array(env_config['scenario_map']),
						number_of_agents_by_team=env_config['number_of_agents_by_team'],
						n_actions_by_team=env_config['n_actions'],
						max_distance_travelled_by_team = env_config['max_distance_travelled_by_team'],
						fleet_initial_positions = env_config['fleet_initial_positions'], #np.array(env_config['fleet_initial_positions']), #
						seed = SEED,
						movement_length_by_team =  env_config['movement_length_by_team'],
						vision_length_by_team = env_config['vision_length_by_team'],
						flag_to_check_collisions_within = env_config['flag_to_check_collisions_within'],
						max_collisions = env_config['max_collisions'],
						reward_function = env_config['reward_function'],
						reward_weights = tuple(env_config['reward_weights']),
						dynamic = env_config['dynamic'],
						obstacles = env_config['obstacles'],
						show_plot_graphics = SHOW_PLOT_GRAPHICS,
						)

f = open(path_to_training_folder + 'experiment_config.json',)
exp_config = json.load(f)
f.close()

network = MultiAgentDuelingDQNAgent(env=env,
						memory_size=int(1E3),  #int(1E6), 1E5
						batch_size=64,
						target_update=1000,
						seed = SEED,
						concensus_actions=exp_config['independent_networks_per_team'],
						device='cuda:0',
						independent_networks_per_team = exp_config['independent_networks_per_team'],
						)

network.load_model(path_to_training_folder + 'BestPolicy.pth')

results = network.evaluate_env(1, render=True)

print(results)

