import sys
import json
sys.path.append('.')

from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Algorithms.LawnMower import LawnMowerAgent
from Algorithms.NRRA import WanderingAgent
from Algorithms.PSO import ParticleSwarmOptimizationFleet
from Algorithms.Greedy import OneStepGreedyFleet
from Algorithms.DRL.ActionMasking.ActionMaskingUtils import ConsensusSafeActionMasking
import numpy as np

algorithms = [
	# 'WanderingAgent', 
    # 'LawnMower', 
    # 'PSO', 
    'Greedy',
	]

SEED = 3
SHOW_RENDER = True
RUNS = 25


# Set config #
scenario_map = np.genfromtxt('Environment/Maps/acoruna_port.csv', delimiter=',')
n_actions_explorers = 8
n_actions_cleaners = 8
n_explorers = 0
n_cleaners = 2
n_agents = n_explorers + n_cleaners
movement_length_explorers = 2
movement_length_cleaners = 1
movement_length_of_each_agent = np.repeat((movement_length_explorers, movement_length_cleaners), (n_explorers, n_cleaners))
vision_length_explorers = 4
vision_length_cleaners = 1
max_distance_travelled_explorers = 400
max_distance_travelled_cleaners = 200
max_steps_per_episode = 150

reward_function = 'backtosimple'
reward_weights=(1, 20, 2, 10)

# Set initial positions #
random_initial_positions = True #coruna_port
if random_initial_positions:
    initial_positions = 'fixed'
else:
    initial_positions = np.array([[32, 7], [30, 7], [28, 7], [26, 7]])[:n_agents, :] # a coruña port
    # initial_positions = None

# Create environment # 
env = MultiAgentCleanupEnvironment(scenario_map = scenario_map,
                        number_of_agents_by_team=(n_explorers,n_cleaners),
                        n_actions_by_team=(n_actions_explorers, n_actions_cleaners),
                        max_distance_travelled_by_team = (max_distance_travelled_explorers, max_distance_travelled_cleaners),
                        max_steps_per_episode = max_steps_per_episode,
                        fleet_initial_positions = initial_positions, # None, 'area', 'fixed' or positions array
                        seed = SEED,
                        movement_length_by_team =  (movement_length_explorers, movement_length_cleaners),
                        vision_length_by_team = (vision_length_explorers, vision_length_cleaners),
                        flag_to_check_collisions_within = False,
                        max_collisions = 1000,
                        reward_function = 'backtosimple', #reward_function,
                        reward_weights = reward_weights,
                        dynamic = True,
                        obstacles = False,
                        show_plot_graphics = SHOW_RENDER,
                        )

for algorithm in algorithms:
    if algorithm == "LawnMower":
        lawn_mower_rng = np.random.default_rng(seed=100)
        agents = [LawnMowerAgent(world=scenario_map, number_of_actions=8, movement_length=movement_length_of_each_agent[i], forward_direction=int(lawn_mower_rng.uniform(0,8)), seed=SEED+i, agent_is_cleaner=env.team_id_of_each_agent[i]==env.cleaners_team_id) for i in range(n_agents)]
    elif algorithm == "WanderingAgent":
        agents = [WanderingAgent(world=scenario_map, number_of_actions=8, movement_length=movement_length_of_each_agent[i], seed=SEED+i, agent_is_cleaner=env.team_id_of_each_agent[i]==env.cleaners_team_id) for i in range(n_agents)]
    elif algorithm == "PSO":
        agents = ParticleSwarmOptimizationFleet(env)
        consensus_safe_masking_module = ConsensusSafeActionMasking(navigation_map = scenario_map, angle_set_of_each_agent=env.angle_set_of_each_agent, movement_length_of_each_agent = env.movement_length_of_each_agent)
    elif algorithm == "Greedy":
        agents = OneStepGreedyFleet(env)

    # Start episodes #
    for run in range(RUNS):
        
        done = {i: False for i in range(n_agents)}
        states = env.reset_env()

        # runtime = 0
        step = 0

        # Reset algorithms #
        if algorithm in ['LawnMower']:
            for i in range(n_agents):
                agents[i].reset(int(lawn_mower_rng.uniform(0,8)) if algorithm == 'LawnMower' else None)
        elif algorithm in ['PSO']:
            agents.reset()
        
        acc_rw_episode = [0 for _ in range(n_agents)]

        while any([not value for value in done.values()]):  # while at least 1 active

            # Add step #
            step += 1
            
            # Take new actions #
            if algorithm  in ['WanderingAgent', 'LawnMower']:
                actions = {agent_id: agents[agent_id].move(actual_position=position, trash_in_pixel=env.model_trash_map[position[0], position[1]]) for agent_id, position in env.get_active_agents_positions_dict().items()}
            elif algorithm == 'PSO':
                q_values = agents.get_agents_actions()
                actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map)
            elif algorithm == 'Greedy':
                actions = agents.get_agents_actions()

            # t0 = time.time()
            states, new_reward, done = env.step(actions)
            acc_rw_episode = [acc_rw_episode[i] + new_reward[i] for i in range(n_agents)]
            # t1 = time.time()
            # runtime += t1-t0

            print(f"Step {env.steps}")
            print(f"Actions: {dict(sorted(actions.items()))}")
            print(f"Rewards: {new_reward}")
            trashes_agents_pixels = {agent_id: env.model_trash_map[position[0], position[1]] for agent_id, position in env.get_active_agents_positions_dict().items()}
            print(f"Trashes in agents pixels: {trashes_agents_pixels}")
            print(f"Trashes removed: {env.trashes_removed_per_agent}")
            print(f"Trashes remaining: {len(env.trash_positions_yx)}")
            print()

        # print('Total runtime: ', runtime)
        print('Total reward: ', acc_rw_episode)
        




average_reward, average_episode_length, mean_cleaned_percentage = network.evaluate_env(RUNS)

for team in range(len(average_reward)):
    print(f'Average reward for team {team}: {average_reward[team]}, with an episode average length of {average_episode_length[team]}. Cleaned percentage: {mean_cleaned_percentage}')