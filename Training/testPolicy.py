import sys
import json
sys.path.append('.')

from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np

path_to_training_folder = 'Training/Trning_RW_backtosimple_1_50_2_10/'
# path_to_training_folder = 'DoneTrainings/Trning_RW_backtosimple_1_20_2_10_20k_curriculum_cleaners/'

f = open(path_to_training_folder + 'environment_config.json',)
env_config = json.load(f)
f.close()

SEED = 3
SHOW_PLOT_GRAPHICS = True
RUNS = 25

env = MultiAgentCleanupEnvironment(scenario_map = np.array(env_config['scenario_map']),
						number_of_agents_by_team=env_config['number_of_agents_by_team'],
						n_actions_by_team=env_config['n_actions'],
						max_distance_travelled_by_team = env_config['max_distance_travelled_by_team'],
						max_steps_per_episode = env_config['max_steps_per_episode'],
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
						concensus_actions=exp_config['concensus_actions'],
						device='cuda:0',
						independent_networks_per_team = exp_config['independent_networks_per_team'],
						curriculum_learning_team=exp_config['curriculum_learning_team'],
						)

# network.load_model(path_to_training_folder + 'Final_Policy.pth')
# network.load_model(path_to_training_folder + 'BestPolicy.pth')
network.load_model(path_to_training_folder + 'BestEvalPolicy.pth')

average_reward, average_episode_length, mean_cleaned_percentage = network.evaluate_env(RUNS)

if exp_config['independent_networks_per_team']:
	for team in range(len(average_reward)):
		print(f'Average reward for team {team}: {average_reward[team]}, with an episode average length of {average_episode_length[team]}. Cleaned percentage: {mean_cleaned_percentage}')
else:
	print(f'Average reward: {average_reward}, with an episode average length of {average_episode_length}. Cleaned percentage: {mean_cleaned_percentage}')