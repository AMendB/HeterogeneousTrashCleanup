import sys
import json
import numpy as np
import os
sys.path.append('.')

# DDQN #
from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent

# PPO #
from Environment.EnvPPOWrapper import EnvWrapper
from Algorithms.PPO.ppo import PPO

path_to_training_folder = 'Training/T//'

f = open(path_to_training_folder + 'environment_config.json',)
env_config = json.load(f)
f.close()

SEED = 3
SHOW_PLOT_GRAPHICS = False
RUNS = 100


if not 'PPO' in path_to_training_folder:
	env = MultiAgentCleanupEnvironment(scenario_map_name = np.array(env_config['scenario_map_name']),
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

	models = [
			'BestEvalPolicy.pth', 
			'BestEvalCleanPolicy.pth', 
			'BestEvalMSEPolicy.pth', 
			'Final_Policy.pth', 
			'BestPolicy.pth'
		    ]
	
	print(f'Algorithm: {path_to_training_folder.split("/")[-2]}\n')
	print(f'Scenario: {env_config["scenario_map_name"]}, with {env_config["number_of_agents_by_team"][0]} explorers and {env_config["number_of_agents_by_team"][1]} cleaners. Dynamic: {env_config["dynamic"]}. Reward function: {env_config["reward_function"]}, Reward weights: {env_config["reward_weights"]}.\n')

	for model in models:

		# Check if the model exists in the folder. If not, skip to the next model
		if model.split('.')[0] in ''.join(os.listdir(path_to_training_folder)):
			network.load_model(path_to_training_folder + model)
		else:	
			continue
		
		env.reset_seeds()
		average_reward, average_episode_length, mean_cleaned_percentage, mean_mse = network.evaluate_env(RUNS)

		print(f'Model: {model}\n')
	
		# if exp_config['independent_networks_per_team']:
		# 	for team in range(len(average_reward)):
		# 		print(f'Average reward for team {team}: {average_reward[team]}, with an episode average length of {average_episode_length[team]}. Cleaned percentage: {round(mean_cleaned_percentage*100,2)}%')
		# else:
		# 	print(f'Average reward: {average_reward}, with an episode average length of {average_episode_length}. Cleaned percentage: {round(mean_cleaned_percentage*100,2)}%')

else:
	env = EnvWrapper(scenario_map_name = np.array(env_config['scenario_map_name']),
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

	# f = open(path_to_training_folder + 'experiment_config.json',)
	# exp_config = json.load(f)
	# f.close()

	state_size = env.observation_space_shape
	action_size = env_config['n_actions'][1]
	n_agents = env_config['number_of_agents_by_team'][1]
	max_steps = env.max_steps_per_episode
	max_samples = 10
	ppo = PPO(env=env, 
			state_size=state_size, 
			action_size=action_size, 
			n_agents=n_agents, 
			max_steps=max_steps, 
			max_samples=max_samples, 
			eval_episodes=RUNS,
			log=False, 
			# logdir=logdir, 
			device='cuda:0')

	model = 'Final_Policy.pth'
	# model = 'BestPolicy.pth'

	ppo.load_model(path_to_training_folder + model)

	acc_rewards_among_agents, cleaned_percentages, n_collisions, _ = map(list, zip(*[ppo.evaluate_env() for _ in range(RUNS)]))

	print(f'Model: {model}')
	print(f'Average reward: {np.mean(acc_rewards_among_agents)}, with a cleaned percentage of {np.mean(cleaned_percentages)*100}%, mean collisions: {np.mean(n_collisions)}')



