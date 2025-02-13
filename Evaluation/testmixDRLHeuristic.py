import sys
import json
import numpy as np
import os
sys.path.append('.')

from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Algorithms.LawnMower import LawnMowerAgent
from Algorithms.NRRA import WanderingAgent
from Algorithms.PSO import ParticleSwarmOptimizationFleet
from Algorithms.Greedy import OneStepGreedyFleet
from Algorithms.DRL.ActionMasking.ActionMaskingUtils import ConsensusSafeActionMasking
import numpy as np
from tqdm import trange

# DDQN #
from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent

path_to_training_folder = 'Training/T/test/'
f = open(path_to_training_folder + 'environment_config.json',)
env_config = json.load(f)
f.close()

algorithms = [
	# 'WanderingAgent', 
    # 'LawnMower', 
    # 'PSO', 
    'Greedy',
	]

model = [
        # 'BestEvalPolicy', 
        # 'BestEvalCleanPolicy', 
        # 'BestEvalMSEPolicy', 
        # 'Final_Policy', 
        # 'BestPolicy',
        'policy',
        ]

SEED = 3
SHOW_RENDER = False
PRINT_INFO = False
RUNS = 100

# Create environment # 
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
                        reward_function = 'negativedijkstra',
                        reward_weights = tuple(env_config['reward_weights']),
                        dynamic = env_config['dynamic'],
                        obstacles = env_config['obstacles'],
                        show_plot_graphics = SHOW_RENDER,
                        )

f = open(path_to_training_folder + 'experiment_config.json',)
exp_config = json.load(f)
f.close()

network = MultiAgentDuelingDQNAgent(env=env,
                        memory_size=int(1E3),  #int(1E6), 1E5
                        batch_size=64,
                        target_update=1000,
                        seed = SEED,
                        consensus_actions=exp_config['concensus_actions'],
                        device='cuda:0',
                        independent_networks_per_team = exp_config['independent_networks_per_team'],
                        curriculum_learning_team=exp_config['curriculum_learning_team'],
                        )
network.load_model(path_to_training_folder + f'{model[0]}.pth')
network.epsilon = 0

for algorithm in algorithms:
    if algorithm == 'LawnMower':
        lawn_mower_rng = np.random.default_rng(seed=100)
        agents = [LawnMowerAgent(world=env.scenario_map, number_of_actions=8, movement_length=env.movement_length_of_each_agent[i], forward_direction=int(lawn_mower_rng.uniform(0,8)), seed=SEED+i, agent_is_cleaner=env.team_id_of_each_agent[i]==env.cleaners_team_id) for i in range(n_agents)]
    elif algorithm == 'WanderingAgent':
        agents = [WanderingAgent(world=env.scenario_map, number_of_actions=8, movement_length=env.movement_length_of_each_agent[i], seed=SEED+i, agent_is_cleaner=env.team_id_of_each_agent[i]==env.cleaners_team_id) for i in range(n_agents)]
    elif algorithm == 'PSO':
        agents = ParticleSwarmOptimizationFleet(env)
        consensus_safe_masking_module = ConsensusSafeActionMasking(navigation_map = env.scenario_map, angle_set_of_each_agent=env.angle_set_of_each_agent, movement_length_of_each_agent = env.movement_length_of_each_agent)
    elif algorithm == 'Greedy':
        agents = OneStepGreedyFleet(env)

    mean_cleaned_percentage = 0
    average_reward = [0 for _ in range(env.n_teams)]
    average_episode_length = [0 for _ in range(env.n_teams)]
    env.reset_seeds()

    # Start episodes #
    for run in trange(RUNS):
        
        done = {i: False for i in range(env.n_agents)}
        states = env.reset_env()

        # runtime = 0
        step = 0

        # Reset algorithms #
        network.nogobackfleet_masking_module.reset()
        if algorithm in ['LawnMower']:
            for i in range(env.n_agents):
                agents[i].reset(int(lawn_mower_rng.uniform(0,8)) if algorithm == 'LawnMower' else None, env.scenario_map)
        elif algorithm in ['WanderingAgent']:
            for i in range(env.n_agents):
                agents[i].reset(env.scenario_map)                
        elif algorithm in ['PSO']:
            agents.reset()
        
        acc_rw_episode = [0 for _ in range(env.n_agents)]
        ep_length_per_teams = [0 for _ in range(env.n_teams)]

        while any([not value for value in done.values()]):  # while at least 1 active

            # Add step #
            step += 1
            
            # Take new actions #
            if algorithm  in ['WanderingAgent', 'LawnMower']:
                actions = {agent_id: agents[agent_id].move(actual_position=position, trash_in_pixel=env.model_trash_map[position[0], position[1]]) for agent_id, position in env.get_active_agents_positions_dict().items()}
            elif algorithm == 'PSO':
                q_values = agents.get_agents_q_values()
                actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map)
            elif algorithm == 'Greedy':
                actions = agents.get_agents_actions()
            # DRL #
            actions = network.select_consensus_actions(states=states, positions=env.get_active_agents_positions_dict(), n_actions_of_each_agent=env.n_actions_of_each_agent, done = done, deterministic=True)

            # Take heuristic actions for explorers and DRL actions for cleaners #
            # actions = {agent_id: actions_drl[agent_id] if env.team_id_of_each_agent[agent_id] == env.explorers_team_id else actions[agent_id] for agent_id in env.get_active_agents_positions_dict().keys()}
            
            # t0 = time.time()
            states, new_reward, done = env.step(actions)
            acc_rw_episode = [acc_rw_episode[i] + new_reward[i] for i in range(env.n_agents)]
            ep_length_per_teams = [ep_length_per_teams[team_id] + 1 if not env.dones_by_teams[team_id] else ep_length_per_teams[team_id] for team_id in env.teams_ids]
            # t1 = time.time()
            # runtime += t1-t0
            
            if PRINT_INFO:
                print(f"Step {env.steps}")
                print(f"Actions: {dict(sorted(actions.items()))}")
                print(f"Rewards: {new_reward}")
                trashes_agents_pixels = {agent_id: env.model_trash_map[position[0], position[1]] for agent_id, position in env.get_active_agents_positions_dict().items()}
                print(f"Trashes in agents pixels: {trashes_agents_pixels}")
                print(f"Trashes removed: {env.trashes_removed_per_agent}")
                print(f"Trashes remaining: {len(env.trash_positions_yx)}")
                print()

        # Print accumulated reward of episode #
        if PRINT_INFO:
            print('Total reward: ', acc_rw_episode)
            # print('Total runtime: ', runtime)
        
        # Calculate mean metrics all episodes #
        mean_cleaned_percentage += env.get_percentage_cleaned_trash()
        for team in range(env.n_teams):
            mean_team_acc_reward_ep = np.mean([acc_rw_episode[i] for i in range(env.n_agents) if env.team_id_of_each_agent[i] == team])
            average_reward[team] += mean_team_acc_reward_ep
            average_episode_length[team] += ep_length_per_teams[team]
    
    # Print algorithm results #
    print(f'Algorithm: {algorithm}. Scenario: {env.scenario_map_name}, with {env.number_of_agents_by_team[0]} explorers and {env.number_of_agents_by_team[1]} cleaners. Dynamic: {env.dynamic}. Dynamic: {env.dynamic}. Reward function: {env.reward_function}, Reward weights: {env.reward_weights}.')

    for team in range(env.n_teams):
        print(f'Average reward for {algorithm} team {team} with {env.number_of_agents_by_team[0] if team==0 else env.number_of_agents_by_team[1]} agents: {average_reward[team]/RUNS}, with an episode average length of {average_episode_length[team]/RUNS}. Cleaned percentage: {round(mean_cleaned_percentage/RUNS*100, 2)}%')
    print()