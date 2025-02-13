import sys
sys.path.append('.')

from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np
import argparse
import optuna
import json

def objective(trial):
	# Hiperparámetros optimizables
	epsilon = trial.suggest_float("epsilon", 0.5, 1.0, step=0.05) # Epsilon upper bound for epsilon-greedy interval
	gamma = trial.suggest_float("gamma", 0.80, 0.99, step=0.01) # Discount factor
	lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
	target_update = trial.suggest_int("target_update", 1000, 10000, step=1000)
	train_every = trial.suggest_int("train_every", 5, 50, step=5)
	batch_size = trial.suggest_int("batch_size", 64, 256, step=32)
	# memory_size = trial.suggest_int("memory_size", 4, 6, step=1)
	# memory_size = int(10**memory_size)


	# Ajustar los pesos de la función de recompensa
	reward_weights = (
		trial.suggest_float("reward_weight_1", 0.1, 10.0),
		trial.suggest_float("reward_weight_2", 1.0, 100.0),
		trial.suggest_float("reward_weight_3", 0.1, 10.0),
		trial.suggest_float("reward_weight_4", 0.1, 5.0)
	)


	# Parámetros fijos
	memory_size = int(1E6)
	reward_function = args.reward_function
	device = args.device
	episodes = args.episodes
	n_explorers = args.n_explorers
	n_cleaners = args.n_cleaners
	max_steps_per_episode = args.max_steps_per_episode
	scenario_map_name = args.scenario_map_name
	vision_length_explorers = 4
	vision_length_cleaners = 1
	max_distance_travelled_explorers = 400
	max_distance_travelled_cleaners = 200
	dynamic_env = True if args.dynamic_env.capitalize() == "True" else False
	obstacles = True if args.obstacles.capitalize() == "True" else False
	greedy_training = True if args.greedy_training.capitalize() == "True" else False
	pso_training = True if args.pso_training.capitalize() == "True" else False
	prewarm_percentage = args.prewarm_percentage

	# Crear entorno
	env = MultiAgentCleanupEnvironment(
		scenario_map_name=scenario_map_name,
		number_of_agents_by_team=(n_explorers, n_cleaners),
		n_actions_by_team=(8, 8),
		max_distance_travelled_by_team=(max_distance_travelled_explorers, max_distance_travelled_cleaners),
		max_steps_per_episode=max_steps_per_episode,
		fleet_initial_positions='fixed',
		seed=0,
		movement_length_by_team=(2, 1),
		vision_length_by_team=(vision_length_explorers, vision_length_cleaners),
		flag_to_check_collisions_within=False,
		max_collisions=10,
		reward_function=reward_function,
		reward_weights=reward_weights,
		dynamic=dynamic_env,
		obstacles=obstacles,
		show_plot_graphics=False,
	)

	if greedy_training:
		training_type = "_greedy"
	elif pso_training:
		training_type = "_pso"
	else:
		training_type = ""
	global folder
	folder = f'optuna_trials_{scenario_map_name}_{reward_function}{training_type}'
	logdir = f'{folder}/trial_{trial.number}'

	# Crear agente
	network = MultiAgentDuelingDQNAgent(
		env=env,
		memory_size=memory_size, 
		batch_size=batch_size,
		target_update=target_update,
		soft_update=False,
		tau=0.001, 
		epsilon_values=[1.0, 0.05],
		epsilon_interval=[0.0, epsilon],
		greedy_training=greedy_training, 
		pso_training=pso_training, 
		learning_starts=100, 
		gamma=gamma,
		lr=lr,
		save_every=10000,
		train_every=train_every,
		masked_actions=False,
		consensus_actions=True,
		device=device,
		logdir=logdir,
		eval_every=50,
		eval_episodes=100,
		prewarm_percentage=prewarm_percentage,
		noisy=False,
		distributional=False,
		independent_networks_per_team=True,
		curriculum_learning_team=None,
	)

	# Entrenar agente durante un número limitado de episodios
	list_of_ptc_evaluations_cleaners = network.train(episodes=episodes, extra_episodes=0, return_feedback=True)

	# Métrica de evaluación: limpieza (PTC) promedio en evaluación
	avg_ptc = np.mean(list_of_ptc_evaluations_cleaners)
	return avg_ptc

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_trials', type=int, default=50, help='Número de pruebas para Optuna.')
	parser.add_argument('--scenario_map_name', type=str, default='acoruna_port', help='Name of the scenario map.')
	parser.add_argument('--n_explorers', type=int, default=2, help='Number of explorers agents.')
	parser.add_argument('--n_cleaners', type=int, default=2, help='Number of cleaners agents.')
	parser.add_argument('--max_steps_per_episode', type=int, default=150, help='Max steps per episode.')
	parser.add_argument('--dynamic_env', type=str, default='True', help='Dynamic environment.')
	parser.add_argument('--obstacles', type=str, default='False', help='Obstacles in the environment.')
	parser.add_argument('-rw', '--reward_function', type=str, default='negativedistance', help='Reward function to use: basic_reward, extended_reward, backtosimple')
	parser.add_argument('-dev', '--device', type=str, default='cuda:0', help='Device to use: cuda:x, cpu')
	parser.add_argument('-eps', '--episodes', type=int, default=1000, help='Number of episodes for each trial.')
	parser.add_argument('-gt', '--greedy_training', type=str, default="False", help='Use greedy training instead of epsilon-greedy training.')
	parser.add_argument('--pso_training', type=str, default="False", help='Use PSO training instead of epsilon-greedy training.')
	parser.add_argument('--prewarm_percentage', type=float, default=0, help='Percentage of memory to prewarm with Greedy actions.')
	args = parser.parse_args()


	# Crear estudio de Optuna
	study = optuna.create_study(direction="maximize", study_name="DQN_hyperparametrization")
	study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

	# Guardar los mejores resultados en un archivo JSON
	best_trial = {
		"value": study.best_trial.value,
		"params": study.best_trial.params
	}
	with open(f"{folder}/best_trial_optuna.json", "w") as f:
		json.dump(best_trial, f, indent=4)

	# Mostrar los mejores resultados
	print("\nBest trial:")
	print(f"  Value: {study.best_trial.value}")
	print("  Params:")
	for key, value in study.best_trial.params.items():
		print(f"    {key}: {value}")
