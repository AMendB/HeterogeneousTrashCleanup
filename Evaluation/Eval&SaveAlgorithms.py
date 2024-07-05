import sys
sys.path.append('.')
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import trange
import pandas as pd
from cycler import cycler

# from Environment.MonitoringEnvironmentPenultimate import MultiAgentMonitoring
from Environment.CleanupEnvironment import MultiAgentCleanupEnvironment
from Evaluation.Utils.metrics_wrapper import MetricsDataCreator


class AlgorithmRecorderAndAnalizer:
    def __init__(self, env:MultiAgentCleanupEnvironment, scenario_map, n_agents,  relative_path, algorithm, reward_funct, reward_weights, runs) -> None:
        self.env = env
        self.scenario_map = scenario_map
        self.n_agents = n_agents
        self.relative_path = relative_path
        self.algorithm = algorithm
        self.reward_funct = reward_funct
        self.reward_weights = reward_weights
        self.runs = runs

    def plot_all_figs(self, run):
        self.env.render()
        plt.show(block=True)
        self.plot_paths(run=run, save_plot=False)
        self.plot_metrics(show_plot=True, run=run, save_plot = False)

    def get_heatmap(self, heatmaps = None, only_save = False):

        if only_save:
            # Save all heatmaps in one figure #
            if len(heatmaps) > 1:
                gridspec = dict(hspace=0.1, width_ratios=[1 + 0.25 * len(heatmaps)]+[1 + 0.25 * len(heatmaps)]*len(heatmaps), height_ratios=[1, 0.3])
                fig, axs = plt.subplots(2, len(heatmaps)+1, figsize=(5* len(heatmaps),6), gridspec_kw=gridspec)

                # HEATMAPS OF ALL AGENTS #
                heatmap_total = np.zeros_like(self.scenario_map)
                for heatmap in heatmaps:
                    heatmap_total += heatmap  
                heatmap_total[self.env.non_water_mask] = np.nan
                axis0 = axs[0][0].imshow(heatmap_total, cmap='YlOrRd', interpolation='nearest')
                axs[0][0].set_title(f"Total")
                fig.colorbar(axis0,ax=axs[0][0], shrink=0.8)
                fig.suptitle(f"{self.algorithm.split('_')[0]} | {self.reward_funct.split('_')[0]}_{'_'.join(map(str, self.reward_weights))}", fontsize=12)
                
                # Percetange visited cells #
                visited = np.count_nonzero(heatmap_total>0)
                visited_5percent = np.count_nonzero(heatmap_total>=self.runs*0.05)
                visited_20percent = np.count_nonzero(heatmap_total>=self.runs*0.20)
                visited_50percent = np.count_nonzero(heatmap_total>=self.runs*0.50)
                visited_80percent = np.count_nonzero(heatmap_total>=self.runs*0.80)
                visitables = len(self.env.visitable_locations)
                axs[1][0].text(0, 0, f"Percentage visited: {round(100*visited/visitables, 2)}%\n"\
                            f"Percentage visited 5% of eps: {round(100*visited_5percent/visitables, 2)}%\n"\
                            f"Percentage visited 20% of eps: {round(100*visited_20percent/visitables, 2)}%\n"\
                            f"Percentage visited 50% of eps: {round(100*visited_50percent/visitables, 2)}%\n"\
                            f"Percentage visited 80% of eps: {round(100*visited_80percent/visitables, 2)}%", 
                            transform = axs[1][0].transAxes, fontsize='small')
                axs[1][0].axis('off')

                # HEATMAPS SPLIT BY TEAM #
                for team, heatmap in enumerate(heatmaps):
                    heatmap[self.env.non_water_mask] = np.nan
                    cax = axs[0][team+1].imshow(heatmap, cmap='YlOrRd', interpolation='nearest')
                    axs[0][team+1].set_title(f"Team: {team}")
                    fig.colorbar(cax,ax=axs[0][team+1], shrink=0.8)
                    axs[1][team+1].axis('off')
                    visited = np.count_nonzero(heatmap>0)
                    visited_5percent = np.count_nonzero(heatmap>=self.runs*0.05)
                    visited_20percent = np.count_nonzero(heatmap>=self.runs*0.20)
                    visited_50percent = np.count_nonzero(heatmap>=self.runs*0.50)
                    visited_80percent = np.count_nonzero(heatmap>=self.runs*0.80)
                    axs[1][team+1].text(0, 0, f"Percentage visited: {round(100*visited/visitables, 2)}%\n"\
                                f"Percentage visited 5% of eps: {round(100*visited_5percent/visitables, 2)}%\n"\
                                f"Percentage visited 20% of eps: {round(100*visited_20percent/visitables, 2)}%\n"\
                                f"Percentage visited 50% of eps: {round(100*visited_50percent/visitables, 2)}%\n"\
                                f"Percentage visited 80% of eps: {round(100*visited_80percent/visitables, 2)}%", 
                                transform = axs[1][team+1].transAxes, fontsize='small')
                
            else:
                gridspec = dict(hspace=0.0, height_ratios=[1, 0.3])
                fig, axs = plt.subplots(2, 1, figsize=(5,6), gridspec_kw=gridspec)
                heatmap_total = heatmaps[0]
                heatmap_total[self.env.non_water_mask] = np.nan
                axis0 = axs[0].imshow(heatmap_total, cmap='YlOrRd', interpolation='nearest')
                axs[0].set_title(f"Total")
                fig.colorbar(axis0,ax=axs[0])
                fig.suptitle(f"{self.algorithm.split('_')[0]} | {self.reward_funct.split('_')[0]}_{'_'.join(map(str, self.reward_weights))}", fontsize=11)
                
                # Percetange visited cells #
                visited = np.count_nonzero(heatmap_total>0)
                visited_5percent = np.count_nonzero(heatmap_total>=self.runs*0.05)
                visited_20percent = np.count_nonzero(heatmap_total>=self.runs*0.20)
                visited_50percent = np.count_nonzero(heatmap_total>=self.runs*0.50)
                visited_80percent = np.count_nonzero(heatmap_total>=self.runs*0.80)
                visitables = len(self.env.visitable_locations)
                axs[1].text(0.25, 0, f"Percentage visited: {round(100*visited/visitables, 2)}%\n"\
                            f"Percentage visited 5% of eps: {round(100*visited_5percent/visitables, 2)}%\n"\
                            f"Percentage visited 20% of eps: {round(100*visited_20percent/visitables, 2)}%\n"\
                            f"Percentage visited 50% of eps: {round(100*visited_50percent/visitables, 2)}%\n"\
                            f"Percentage visited 80% of eps: {round(100*visited_80percent/visitables, 2)}%", 
                            transform = axs[1].transAxes, fontsize='small')
                axs[1].axis('off')
                
            plt.savefig(fname=f"{self.relative_path}/Heatmaps.png")
            plt.close()
            

        else:
            if heatmaps is None:
                heatmaps = [np.zeros_like(self.scenario_map) for _ in range(self.env.n_teams)]
            
            # Heatmap by team #
            for team in range(self.env.n_teams):           
                visited_locations = np.vstack([self.env.fleet.vehicles[agent].waypoints for agent in range(self.n_agents) if self.env.team_id_of_each_agent[agent] == team])
                heatmaps[team][visited_locations[:,0], visited_locations[:,1]] += 1

        return(heatmaps)

    def plot_paths(self, run = None, save_plot = False):
        # Agents path plot over ground truth #
        plt.figure(figsize=(7,5))

        waypoints = [self.env.fleet.vehicles[i].waypoints for i in range(self.n_agents)]
        gt = self.env.real_trash_map.copy()
        gt[self.env.non_water_mask] = np.nan

        plt.imshow(gt,  cmap ='cet_linear_bgy_10_95_c74', vmin = 0.0, vmax = 1.0, alpha = 0.3)

        for agent, agents_waypoints in enumerate(waypoints):
            y = [point[0] for point in agents_waypoints]
            x = [point[1] for point in agents_waypoints]
            plt.plot(x, y, color=self.env.colors_agents[agent+2])


        if save_plot:
            plt.title(f"{self.algorithm.split('_')[0]} | {self.reward_funct.split('_')[0]}_{'_'.join(map(str, self.reward_weights))} | EP{run}", fontsize='medium')
            plt.savefig(fname=f"{self.relative_path}/Paths/Ep{run}.png")
            plt.savefig(fname=f"{self.relative_path}/Paths_svg/Ep{run}.svg")
            plt.close()
        else:
            plt.title(f"Real Importance (GT) with agents path, EP {run}")
            plt.show(block=True)  # Mostrar el gráfico resultante

    def plot_metrics(self, show_plot=False, run = None, save_plot = False, plot_std=False):
            # Reward and Error final graphs #
            fig, ax = plt.subplots(2, 3, figsize=(17,9))

            ax[0][0].set_prop_cycle(cycler(color=self.env.colors_agents[-self.n_agents:]))
            ax[0][0].plot(self.reward_agents_acc, '-')
            ax[0][0].set(title = 'Reward', xlabel = 'Step', ylabel = 'Individual Reward')
            ax[0][0].legend([f'Agent {i}' for i in range(self.n_agents)])
            ax[0][0].plot(self.reward_acc, 'b-', linewidth=4)
            ax[0][0].grid()
            if plot_std:
                for agent in range(self.n_agents):
                    ax[0][0].fill_between(self.results_std.index, self.results_mean[f'AccRw{agent}'] - self.results_std[f'AccRw{agent}'], self.results_mean[f'AccRw{agent}'] + self.results_std[f'AccRw{agent}'], alpha=0.2, label='Std')
                ax[0][0].fill_between(self.results_std.index, self.results_mean['R_acc'] - self.results_std['R_acc'], self.results_mean['R_acc'] + self.results_std['R_acc'], color='b', alpha=0.2, label='Std')

            ax[0][1].plot(self.mse, '-', label='Media')
            # ax[0][1].plot(self.mse_peaks, '-')
            ax[0][1].set(title = 'MSE', xlabel = 'Step')
            # ax[0][1].legend(['Total', 'In Peaks'])
            ax[0][1].grid()
            if plot_std:
                ax[0][1].fill_between(self.results_std.index, self.results_mean['MSE'] - self.results_std['MSE'], self.results_mean['MSE'] + self.results_std['MSE'], alpha=0.2, label='Std')

            # ax[0][2].plot(self.mse_peaks, '-')
            # ax[0][2].set(title = 'MSE_peaks', xlabel = 'Step')
            # ax[0][2].grid()
            # if plot_std:
            #     ax[0][2].fill_between(self.results_std.index, self.results_mean['MSE_peaks'] - self.results_std['MSE_peaks'], self.results_mean['MSE_peaks'] + self.results_std['MSE_peaks'], alpha=0.2, label='Std')

            # ax[1][0].plot(self.uncert_mean, '-')
            # ax[1][0].plot(self.uncert_max, '-')
            # ax[1][0].set(title = 'Uncertainty', xlabel = 'Step')
            # ax[1][0].legend(['Mean', 'Max'])
            # ax[1][0].grid()
            # if plot_std:
            #     ax[1][0].fill_between(self.results_std.index, self.results_mean['Uncert_mean'] - self.results_std['Uncert_mean'], self.results_mean['Uncert_mean'] + self.results_std['Uncert_mean'], alpha=0.2, label='Std')
            #     ax[1][0].fill_between(self.results_std.index, self.results_mean['Uncert_max'] - self.results_std['Uncert_max'], self.results_mean['Uncert_max'] + self.results_std['Uncert_max'], alpha=0.2, label='Std')

            plot_traveled_distance = False
            if plot_traveled_distance == True or self.n_agents == 1:
                ax[1][1].bar(list(map(str, [*range(self.n_agents)])) ,self.traveled_distance_agents[-1], width=0.4, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
                ax[1][1].set(title = 'Traveled_distance', xlabel = 'Agent')
                ax[1][1].set_ylim(90, 110)
                ax[1][1].grid()
            else:
                mean_distance = np.mean(self.distances_between_agents, axis=1)
                ax[1][1].plot(mean_distance, '-')
                ax[1][1].set(title = 'Mean distance between agents', xlabel = 'Step', ylabel = 'Distance')
                # ax[1][1].legend([*env.fleet.get_distances_between_agents().keys()])
                ax[1][1].grid()

            ax[1][2].plot(self.max_redundancy, '-')
            ax[1][2].set(title = 'Max_Redundancy', xlabel = 'Step')
            ax[1][2].grid()
            if plot_std:
                ax[1][2].fill_between(self.results_std.index, self.results_mean['Max_Redundancy'] - self.results_std['Max_Redundancy'], self.results_mean['Max_Redundancy'] + self.results_std['Max_Redundancy'], alpha=0.2, label='Std')

            if not(run is None):
                fig.suptitle(f'Run nº: {run}. Alg: {self.algorithm} Rw: {self.reward_funct}_' + '_'.join(map(str, reward_weights)), fontsize=16)
            else:
                fig.suptitle(f'{len(self.runs)} episodes. {self.algorithm} | {self.reward_funct}_' + '_'.join(map(str, reward_weights)), fontsize=16)

            if save_plot:
                fig.savefig(fname=f"{self.relative_path}/AverageMetrics_{len(self.runs)}eps.png")
                fig.savefig(fname=f"{self.relative_path}/AverageMetrics_{len(self.runs)}eps.svg")

            if show_plot:
                plt.show(block=True)

    # def save_ground_truths(self, ground_truths_to_save):

    #     output_filename = '/GroundTruths.npy'

    #     np.save( self.relative_path + output_filename , ground_truths_to_save)

    def save_registers(self, new_reward=None, reset=False):

        # Get data # 
        if reset == True:
            self.reward_steps_agents = [[0 for _ in range(self.n_agents)]]
            self.reward_steps = [0]
            self.mse = [self.env.get_model_mse()]
            self.sum_model_changes = [0]
            self.trash_remaining = [len(self.env.trash_positions_yx)]
            self.percentage_of_trash_collected = [0]
            self.traveled_distance_agents = [self.env.fleet.get_fleet_distances_traveled()]
            self.traveled_distance = [0]
            self.max_redundancy = [self.env.get_redundancy_max()]
            if self.n_agents > 1:
                self.distances_between_agents = []
        else:
            # Add new metrics data #
            self.reward_steps_agents.append(list(new_reward.values()))
            self.reward_steps.append(np.sum(list(new_reward.values())))
            self.mse.append(self.env.get_model_mse())
            self.sum_model_changes.append(self.sum_model_changes[-1] + self.env.get_changes_in_model())
            self.trash_remaining.append(len(self.env.trash_positions_yx))
            self.percentage_of_trash_collected.append( 100 * (1 - self.trash_remaining[-1] / self.trash_remaining[0]) )
            self.traveled_distance_agents.append(self.env.fleet.get_fleet_distances_traveled())
            self.traveled_distance.append(np.sum(self.env.fleet.get_fleet_distances_traveled()))
            self.max_redundancy.append(self.env.get_redundancy_max())

        self.reward_acc = np.cumsum(self.reward_steps)
        self.reward_agents_acc = np.cumsum(self.reward_steps_agents, axis=0)
        if self.n_agents > 1:
            self.distances_between_agents.append([*self.env.fleet.get_distances_between_agents().values()])

        # Save metrics #
        if self.n_agents > 1:
            data = [*self.reward_agents_acc[-1], self.reward_acc[-1], self.mse[-1], self.sum_model_changes[-1], self.trash_remaining[-1], self.percentage_of_trash_collected[-1], self.traveled_distance[-1], self.max_redundancy[-1], *self.traveled_distance_agents[-1], *self.distances_between_agents[-1]]
        else:
            data = [*self.reward_agents_acc[-1], self.reward_acc[-1], self.mse[-1], self.sum_model_changes[-1], self.trash_remaining[-1], self.percentage_of_trash_collected[-1], self.traveled_distance[-1], self.max_redundancy[-1], *self.traveled_distance_agents[-1]]
        metrics.save_step(run_num=run, step=step, metrics=data)

        # Save waypoints #
        for veh_id, veh in enumerate(self.env.fleet.vehicles):
            waypoints.save_step(run_num=run, step=step, metrics=[veh_id, veh.actual_agent_position[0], veh.actual_agent_position[1], done[veh_id]])

    def plot_and_tables_metrics_average(self, metrics_path, table, wilcoxon_dict, show_plot = True , save_plot = False):
 
        metrics_df = MetricsDataCreator.load_csv_as_df(metrics_path)
        self.runs = metrics_df['Run'].unique()

        # Obtain dataframes #
        numeric_columns = metrics_df.select_dtypes(include=[np.number])
        self.results_mean = numeric_columns.groupby('Step').agg('mean')
        self.results_std = numeric_columns.groupby('Step').agg('std')

        # Extract data to plot or save fig metrics #
        if show_plot or save_plot:
            first_accreward_agent_index = self.results_mean.columns.get_loc('AccRw0')
            self.reward_agents_acc = self.results_mean.iloc[:, first_accreward_agent_index:first_accreward_agent_index + self.n_agents].values.tolist()
            self.reward_acc = self.results_mean['R_acc'].values.tolist()
            self.mse = self.results_mean['MSE'].values.tolist()
            self.mse_std = self.results_std['MSE'].values.tolist()
            self.sum_model_changes = self.results_mean['Sum_model_changes'].values.tolist()
            self.trash_remaining = self.results_mean['Trash_remaining'].values.tolist()
            self.percentage_of_trash_collected = self.results_mean['Percentage_of_trash_collected'].values.tolist()
            self.traveled_distance = self.results_mean['Traveled_distance'].values.tolist()
            self.max_redundancy = self.results_mean['Max_Redundancy'].values.tolist()
            first_traveldist_agent_index = self.results_mean.columns.get_loc('TravelDist0')
            self.traveled_distance_agents = self.results_mean.iloc[:, first_traveldist_agent_index:first_traveldist_agent_index + self.n_agents].values.tolist()
            if self.n_agents > 1:
                first_distbetween_index = self.results_mean.columns.get_loc([*env.fleet.get_distances_between_agents().keys()][0])
                self.distances_between_agents = self.results_mean.iloc[:, first_distbetween_index:first_distbetween_index + int((self.n_agents*(self.n_agents-1))/2)].values.tolist()

            self.plot_metrics(show_plot=show_plot, save_plot=save_plot, plot_std=True)
            plt.close('all')

        # LATEX TABLE #
        import warnings
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

        name_alg = self.algorithm.split('_')[0].capitalize()
        name_rw = self.reward_funct.split('_')[0].capitalize() + '-' + '-'.join(map(str, reward_weights))

        if not name_alg.capitalize() in table:
            new_df = pd.DataFrame(
                    columns=pd.MultiIndex.from_product([[name_alg], ["Mean33", "CI33 95%", "Mean66", "CI66 95%", "Mean100", "CI100 95%"]]),
                    index=table.index)
            table = pd.concat([table, new_df], axis=1)

        # Calculate the some metrics at 33%, 66% and 100% of each episode and add to table #
        mse = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['MSE']) for percentage in [33, 66, 100]}
        smc = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['Sum_model_changes']) for percentage in [33, 66, 100]}
        ptc = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['Percentage_of_trash_collected']) for percentage in [33, 66, 100]}
        accrw0 = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['AccRw0']) for percentage in [33, 66, 100]}
        accrw1 = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['AccRw1']) for percentage in [33, 66, 100]}
        accrw2 = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['AccRw2']) for percentage in [33, 66, 100]}
        accrw3 = {f'{percentage}%': np.array(numeric_columns[numeric_columns['Step']==round(self.env.max_steps_per_episode*percentage/100)]['AccRw3']) for percentage in [33, 66, 100]}

        table.loc['MSE-'+name_rw, name_alg] = [np.mean(mse['33%']), 1.96*np.std(mse['33%'])/np.sqrt(len(self.runs)), np.mean(mse['66%']),1.96*np.std(mse['66%'])/np.sqrt(len(self.runs)), np.mean(mse['100%']), 1.96*np.std(mse['100%'])/np.sqrt(len(self.runs))]
        table.loc['SumModelChanges-'+name_rw, name_alg] = [np.mean(smc['33%']), 1.96*np.std(smc['33%'])/np.sqrt(len(self.runs)), np.mean(smc['66%']),1.96*np.std(smc['66%'])/np.sqrt(len(self.runs)), np.mean(smc['100%']), 1.96*np.std(smc['100%'])/np.sqrt(len(self.runs))]
        table.loc['PercentageOfTrashCollected-'+name_rw, name_alg] = [np.mean(ptc['33%']), 1.96*np.std(ptc['33%'])/np.sqrt(len(self.runs)), np.mean(ptc['66%']),1.96*np.std(ptc['66%'])/np.sqrt(len(self.runs)), np.mean(ptc['100%']), 1.96*np.std(ptc['100%'])/np.sqrt(len(self.runs))]
        table.loc['AccumulatedReward0-'+name_rw, name_alg] = [np.mean(accrw0['33%']), 1.96*np.std(accrw0['33%'])/np.sqrt(len(self.runs)), np.mean(accrw0['66%']),1.96*np.std(accrw0['66%'])/np.sqrt(len(self.runs)), np.mean(accrw0['100%']), 1.96*np.std(accrw0['100%'])/np.sqrt(len(self.runs))]
        table.loc['AccumulatedReward1-'+name_rw, name_alg] = [np.mean(accrw1['33%']), 1.96*np.std(accrw1['33%'])/np.sqrt(len(self.runs)), np.mean(accrw1['66%']),1.96*np.std(accrw1['66%'])/np.sqrt(len(self.runs)), np.mean(accrw1['100%']), 1.96*np.std(accrw1['100%'])/np.sqrt(len(self.runs))]
        table.loc['AccumulatedReward2-'+name_rw, name_alg] = [np.mean(accrw2['33%']), 1.96*np.std(accrw2['33%'])/np.sqrt(len(self.runs)), np.mean(accrw2['66%']),1.96*np.std(accrw2['66%'])/np.sqrt(len(self.runs)), np.mean(accrw2['100%']), 1.96*np.std(accrw2['100%'])/np.sqrt(len(self.runs))]
        table.loc['AccumulatedReward3-'+name_rw, name_alg] = [np.mean(accrw3['33%']), 1.96*np.std(accrw3['33%'])/np.sqrt(len(self.runs)), np.mean(accrw3['66%']),1.96*np.std(accrw3['66%'])/np.sqrt(len(self.runs)), np.mean(accrw3['100%']), 1.96*np.std(accrw3['100%'])/np.sqrt(len(self.runs))]
        

        # To do WILCOXON TEST, extract the MSE vector of len(vector)=runs from df at steps 33%, 66% and 100% of n_steps_per_episode for each episode #
        n_steps_per_episode = self.env.max_steps_per_episode
        series_33 = numeric_columns[numeric_columns['Step']==round(n_steps_per_episode*0.33)]['MSE'].reset_index(drop='True').rename('33')
        series_66 = numeric_columns[numeric_columns['Step']==round(n_steps_per_episode*0.66)]['MSE'].reset_index(drop='True').rename('66')
        series_100 = numeric_columns[numeric_columns['Step']==round(n_steps_per_episode*1)]['MSE'].reset_index(drop='True').rename('100')
        wilcoxon_dict[f'{name_alg.capitalize()} - {name_rw.capitalize()}'] = pd.concat([series_33, series_66, series_100], axis=1)
        
        return table, wilcoxon_dict


def wilcoxon_test(wilcoxon_dict):
        from itertools import combinations, product
        from scipy.stats import wilcoxon

        results = {}
        
        metrics = ["33", "66", "100"]
        
        for metric in metrics:
            for alg1, alg2 in combinations(wilcoxon_dict.keys(), 2):
                data1 = wilcoxon_dict[alg1]
                data2 = wilcoxon_dict[alg2]

                for metric in metrics:
                    
                    statistic, p_value = wilcoxon(data1[f'{metric}'], data2[f'{metric}'])
                    
                    key = f"{alg1} vs {alg2} - {metric}"
                    results[key] = {
                        "Statistic": statistic,
                        "P-Value": p_value,
                        "Significant": p_value < 0.05  # Puedes ajustar el nivel de significancia aquí
                    }
        
        return results
        

if __name__ == '__main__':

    import time
    from Algorithms.LawnMower import LawnMowerAgent
    from Algorithms.NRRA import WanderingAgent
    from Algorithms.PSO import ParticleSwarmOptimizationFleet
    from Algorithms.Greedy import OneStepGreedyFleet
    from Algorithms.DRL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
    from Algorithms.DRL.ActionMasking.ActionMaskingUtils import ConsensusSafeActionMasking

    algorithms = [
        # 'WanderingAgent', 
        # 'LawnMower', 
        # 'PSO', 
        # 'Greedy',
        # 'Training/Trning_RW_basic_10_10_0/',
        # 'DoneTrainings/Trning_RW_basic_10_10_0 (buffer float16)/',
        # 'DoneTrainings/Trning_RW_basic_10_10_0 (sin penalización)/',
        # 'DoneTrainings/Trning_RW_extended_10_50_0/',
        # 'DoneTrainings/Trning_RW_extended_5_100_0/',
        # 'DoneTrainings/Trning_RW_backtosimple_1_10_2/',
        'DoneTrainings/Trning_RW_backtosimple_1_10_2_10/',
        'DoneTrainings/Trning_RW_backtosimple_1_100_2_10/',
        ]

    SHOW_RENDER = False
    SHOW_FINAL_EP_PLOT = False
    SHOW_FINAL_EVALUATION_PLOT = False

    SAVE_PLOTS_OF_METRICS_AND_PATHS = False
    SAVE_COLLAGES = False
    SAVE_DATA = True
    SAVE_WILCOX = False

    RUNS = 100
    SEED = 3

    EXTRA_NAME = 'Final_Policy'
    # EXTRA_NAME = 'BestPolicy'
    EXTRA_NAME = 'BestEvalPolicy'




    saving_paths = []
    data_table_average = pd.DataFrame() 
    wilcoxon_dict = {}              

    for path_to_training_folder in algorithms:

        if path_to_training_folder in ['WanderingAgent', 'LawnMower', 'PSO', 'Greedy']:
            selected_algorithm = path_to_training_folder

            # Set config #
            # scenario_map = np.genfromtxt('Environment/Maps/ypacarai_map_low_res.csv', delimiter=',')
            scenario_map = np.genfromtxt('Environment/Maps/acoruna_port.csv', delimiter=',') #coruna_port
            n_actions_explorers = 9
            n_actions_cleaners = 10
            n_explorers = 2
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

            reward_function = 'backtosimple' # 'basic_reward', 'extended_reward', 'backtosimple'
            reward_weights=(1, 10, 2)

            # Set initial positions #
            random_initial_positions = True #coruna_port
            if random_initial_positions:
                initial_positions = 'fixed'
            else:
                # initial_positions = np.array([[30, 20], [40, 25], [40, 20], [30, 28]])[:n_agents, :] # ypacarai lake
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
                                    reward_function = reward_function,
                                    reward_weights = reward_weights,
                                    dynamic = True,
                                    obstacles = False,
                                    show_plot_graphics = SHOW_RENDER,
                                    )
            
            if selected_algorithm == "LawnMower":
                lawn_mower_rng = np.random.default_rng(seed=100)
                selected_algorithm_agents = [LawnMowerAgent(world=scenario_map, number_of_actions=8, movement_length=movement_length_of_each_agent[i], forward_direction=int(lawn_mower_rng.uniform(0,8)), seed=SEED+i, agent_is_cleaner=env.team_id_of_each_agent[i]==env.cleaners_team_id) for i in range(n_agents)]
            elif selected_algorithm == "WanderingAgent":
                selected_algorithm_agents = [WanderingAgent(world=scenario_map, number_of_actions=8, movement_length=movement_length_of_each_agent[i], seed=SEED+i, agent_is_cleaner=env.team_id_of_each_agent[i]==env.cleaners_team_id) for i in range(n_agents)]
            elif selected_algorithm == "PSO":
                selected_algorithm_agents = ParticleSwarmOptimizationFleet(env)
                consensus_safe_masking_module = ConsensusSafeActionMasking(navigation_map = scenario_map, angle_set_of_each_agent=env.angle_set_of_each_agent, movement_length_of_each_agent = env.movement_length_of_each_agent)
            elif selected_algorithm == "Greedy":
                selected_algorithm_agents = OneStepGreedyFleet(env)

        else:
            # Load env config #
            f = open(path_to_training_folder + 'environment_config.json',)
            env_config = json.load(f)
            f.close()
            
            env = MultiAgentCleanupEnvironment(scenario_map = np.array(env_config['scenario_map']),
                                    number_of_agents_by_team=env_config['number_of_agents_by_team'],
                                    n_actions_by_team=env_config['n_actions'],
                                    max_distance_travelled_by_team = env_config['max_distance_travelled_by_team'],
                                    max_steps_per_episode=env_config['max_steps_per_episode'],
                                    fleet_initial_positions = env_config['fleet_initial_positions'], # np.array(env_config['fleet_initial_positions']), #
                                    seed = SEED,
                                    movement_length_by_team =  env_config['movement_length_by_team'],
                                    vision_length_by_team = env_config['vision_length_by_team'],
                                    flag_to_check_collisions_within = env_config['flag_to_check_collisions_within'],
                                    max_collisions = env_config['max_collisions'],
                                    reward_function = env_config['reward_function'],
                                    reward_weights = tuple(env_config['reward_weights']),
                                    dynamic = env_config['dynamic'],
                                    obstacles = env_config['obstacles'],
                                    show_plot_graphics = SHOW_RENDER,
                                    )
            scenario_map = env.scenario_map
            n_agents = env.n_agents
            reward_function = env.reward_function
            reward_weights = env.reward_weights
            
            # Load exp config #
            f = open(path_to_training_folder + 'experiment_config.json',)
            exp_config = json.load(f)
            f.close()

            independent_networks_per_team = exp_config['independent_networks_per_team']

            if independent_networks_per_team:
                selected_algorithm = "Independent_Networks_Per_Team"
            else:
                selected_algorithm = "Network"
                # raise NotImplementedError("This algorithm is not implemented. Choose one that is.")

            network = MultiAgentDuelingDQNAgent(env=env,
                                    memory_size=int(1E3),  #int(1E6), 1E5
                                    batch_size=exp_config['batch_size'],
                                    target_update=1000,
                                    seed = SEED,
                                    concensus_actions=exp_config['concensus_actions'],
                                    device='cuda:0',
                                    independent_networks_per_team = independent_networks_per_team,
                                    )
            network.load_model(path_to_training_folder + f'{EXTRA_NAME}.pth')
            network.epsilon = 0

        if SAVE_DATA:
            # Reward function and create path to save #
            relative_path = f'Evaluation/Results/{EXTRA_NAME}' + selected_algorithm.split('_')[0] + '.' + str(n_agents) + '.' + reward_function.split('_')[0] + '_' + '_'.join(map(str, reward_weights))
            if not(os.path.exists(relative_path)): # create the directory if not exists
                os.mkdir(relative_path)
                os.mkdir(f'{relative_path}/Paths')
                os.mkdir(f'{relative_path}/Paths_svg')
                # os.mkdir(f'{relative_path}/Models')
                # os.mkdir(f'{relative_path}/Models_svg')
            saving_paths.append(relative_path)

            # algorithm_analizer = AlgorithmRecorderAndAnalizer(env, scenario_map, n_agents, relative_path, selected_algorithm, reward_function, reward_weights)
            algorithm_analizer = AlgorithmRecorderAndAnalizer(env, scenario_map, n_agents, relative_path, selected_algorithm, f'{EXTRA_NAME}{reward_function}', reward_weights, RUNS)
            env.save_environment_configuration(relative_path)

            # Initialize metrics saving class #
            metrics_names = [*[f'AccRw{id}' for id in range(n_agents)], 'R_acc', 'MSE', 'Sum_model_changes', 'Trash_remaining', 'Percentage_of_trash_collected',
                            'Traveled_distance', 'Max_Redundancy', *[f'TravelDist{id}' for id in range(n_agents)], *env.fleet.get_distances_between_agents().keys()]
            metrics = MetricsDataCreator(metrics_names=metrics_names,
                                        algorithm_name=selected_algorithm + '.' + str(n_agents) + '.' + reward_function + '.' + '_'.join(map(str, reward_weights)),
                                        experiment_name= 'metrics',
                                        directory=relative_path )

            waypoints = MetricsDataCreator(metrics_names=['vehicle', 'x', 'y', 'Done'],
                                        algorithm_name=selected_algorithm + '.' + str(n_agents) + '.' + reward_function + '.' + '_'.join(map(str, reward_weights)),
                                        experiment_name= 'waypoints',
                                        directory=relative_path)
            
            # ground_truths_to_save = []
            heatmaps = None

        # Start episodes #
        for run in trange(RUNS):
            
            done = {i: False for i in range(n_agents)}
            states = env.reset_env()

            # runtime = 0
            step = 0

            # Save data #
            if SAVE_DATA:
                algorithm_analizer.save_registers(reset=True)
                # ground_truths_to_save.append(env.real_trash_map)
            
            if selected_algorithm in ['LawnMower']:
                for i in range(n_agents):
                    # selected_algorithm_agents[i].reset(0)
                    selected_algorithm_agents[i].reset(int(lawn_mower_rng.uniform(0,8)) if selected_algorithm == 'LawnMower' else None)
            elif selected_algorithm in ['PSO','Greedy']:
                selected_algorithm_agents.reset()

            # Take first actions #
            if selected_algorithm  in ['Independent_Networks_Per_Team', 'Network']:
                network.nogobackfleet_masking_module.reset()
                actions = network.select_concensus_actions(states=states, positions=env.get_active_agents_positions_dict(), n_actions_of_each_agent=env.n_actions_of_each_agent, done = done, deterministic=True)
            elif selected_algorithm  in ['WanderingAgent', 'LawnMower']:
                actions = {agent_id: selected_algorithm_agents[agent_id].move(actual_position=position, trash_in_pixel=env.model_trash_map[position[0], position[1]]) for agent_id, position in env.get_active_agents_positions_dict().items()}
            elif selected_algorithm == 'PSO':
                q_values = selected_algorithm_agents.get_agents_actions()
                actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map)
            elif selected_algorithm == 'Greedy':
                actions = selected_algorithm_agents.get_agents_actions()
            
            acc_rw_episode = [0 for _ in range(n_agents)]
            while any([not value for value in done.values()]):  # while at least 1 active
                
                step += 1

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

                # Save data #
                if SAVE_DATA:
                    algorithm_analizer.save_registers(new_reward, reset=False)

                # Take new actions #
                if selected_algorithm  in ['Independent_Networks_Per_Team', 'Network']:
                    actions = network.select_concensus_actions(states=states, positions=env.get_active_agents_positions_dict(), n_actions_of_each_agent=env.n_actions_of_each_agent, done = done, deterministic=True)
                elif selected_algorithm  in ['WanderingAgent', 'LawnMower']:
                    actions = {agent_id: selected_algorithm_agents[agent_id].move(actual_position=position, trash_in_pixel=env.model_trash_map[position[0], position[1]]) for agent_id, position in env.get_active_agents_positions_dict().items()}
                elif selected_algorithm == 'PSO':
                    q_values = selected_algorithm_agents.get_agents_actions()
                    actions = consensus_safe_masking_module.query_actions(q_values=q_values, agents_positions=env.get_active_agents_positions_dict(), model_trash_map=env.model_trash_map)
                elif selected_algorithm == 'Greedy':
                    actions = selected_algorithm_agents.get_agents_actions()

            # print('Total runtime: ', runtime)
            print('Total reward: ', acc_rw_episode)
            
            if SHOW_FINAL_EP_PLOT:
                algorithm_analizer.plot_all_figs(run)
            
            if SAVE_PLOTS_OF_METRICS_AND_PATHS and run%1 == 0:
                algorithm_analizer.plot_paths(run, save_plot=SAVE_PLOTS_OF_METRICS_AND_PATHS)
            
            if SAVE_DATA:
                heatmaps = algorithm_analizer.get_heatmap(heatmaps)

        if SAVE_DATA:
            # algorithm_analizer.save_ground_truths(ground_truths_to_save)
            algorithm_analizer.get_heatmap(heatmaps, only_save=True)
            metrics.save_data_as_csv()
            waypoints.save_data_as_csv()

        data_table_average, wilcoxon_dict = algorithm_analizer.plot_and_tables_metrics_average(metrics_path=relative_path + '/metrics.csv', 
                                                                                               table=data_table_average, wilcoxon_dict=wilcoxon_dict, 
                                                                                               show_plot=SHOW_FINAL_EVALUATION_PLOT,save_plot=SAVE_PLOTS_OF_METRICS_AND_PATHS)
    
    if SAVE_DATA:
        if len(algorithms) > 1:
            with open(f'Evaluation/Results/{EXTRA_NAME}LatexTableAverage{RUNS}eps_{n_agents}A.txt', "w") as f:
                f.write(data_table_average.style.to_latex())
            with open(f'Evaluation/Results/{EXTRA_NAME}TableAverage{RUNS}eps_{n_agents}A.txt', "w") as f:
                print(data_table_average.to_markdown(), file=f)
        else:
            with open(f'{relative_path}/{EXTRA_NAME}LatexTableAverage{RUNS}eps_{n_agents}A.txt', "w") as f:
                f.write(data_table_average.style.to_latex())
            with open(f'{relative_path}/{EXTRA_NAME}TableAverage{RUNS}eps_{n_agents}A.txt', "w") as f:
                print(data_table_average.to_markdown(), file=f)

        # Test de Wilcoxon # 
        if SAVE_WILCOX:
            results = wilcoxon_test(wilcoxon_dict)
            file = open(f'Evaluation/Results/{EXTRA_NAME}Wilcoxon{RUNS}eps_{n_agents}A.txt', "w")
            for key, result in results.items():
                info = f"Test de Wilcoxon para {key}: Estadístico = {result['Statistic']}, Valor p = {result['P-Value']}, Significativo = {result['Significant']}"
                print(info)
                file.write(info + '\n')
            file.close()
        

        if SAVE_COLLAGES:
            from shutil import rmtree
            import cv2

            # Function to crop an image
            def crop_image(image, x, y, width, high):
                return image[y:y+high, x:x+width]
            
            # Collage agents paths #
            images_paths = [sorted([os.path.join(f'{path}/Paths/', file) for file in os.listdir(f'{path}/Paths/')], key=lambda x: int(x.split('/')[-1].replace('Ep', '').replace('.png', ''))) for path in saving_paths]
            collage = np.hstack([np.vstack([crop_image(cv2.imread(img), 70, 20, 580, 485) for img in algorithm]) for algorithm in images_paths])
            cv2.imwrite(f'Evaluation/Results/{EXTRA_NAME}Paths{RUNS}eps_{n_agents}A.png', collage)
            
            # collage GP models #
            # images_paths = [sorted([os.path.join(f'{path}/Models/', file) for file in os.listdir(f'{path}/Models/')], key=lambda x: int(x.split('/')[-1].replace('Ep', '').replace('.png', ''))) for path in saving_paths]
            # gt_collage = np.vstack([crop_image(cv2.imread(img), 0, 0, 500, 2000) for img in images_paths[-1]])
            # collage = np.hstack([np.vstack([crop_image(cv2.imread(img), 500, 0, 700, 2000) for img in algorithm]) for algorithm in images_paths])
            # collage = np.hstack([gt_collage, collage])
            # cv2.imwrite(f'Evaluation/Results/{EXTRA_NAME}Models{RUNS}eps_{n_agents}A.png', collage)

            # Collage average metrics #
            networks = set([path.split('/')[-2].split('.')[0] for path in saving_paths])
            collage = []
            for net in networks:
                algorithms_paths = [path for path in saving_paths if net in path]
                images_paths = [next(os.path.join(path, file) for file in os.listdir(path) if file.startswith('AverageMetrics') and file.endswith('png')) for path in algorithms_paths]
                collage.append(np.hstack([crop_image(cv2.imread(img), 100, 0, 1580, 880) for img in images_paths]))
            collage = np.vstack(collage)
            cv2.imwrite(f'Evaluation/Results/{EXTRA_NAME}MetricsAverage{RUNS}eps_{n_agents}A.png', collage)

            # Collage average heatmaps #
            collage = []
            for net in networks:
                algorithms_paths = [path for path in saving_paths if net in path]
                images_paths = [next(os.path.join(path, file) for file in os.listdir(path) if file.startswith('Heatmap') and file.endswith('png')) for path in algorithms_paths]
                collage.append(np.hstack([crop_image(cv2.imread(img), 0, 0, 2000, 2000) for img in images_paths]))
            collage = np.vstack(collage)
            cv2.imwrite(f'Evaluation/Results/{EXTRA_NAME}HeatmapsAverage{RUNS}eps_{n_agents}A.png', collage)

            # Remove Paths and Models folders #
            for path in saving_paths:
                rmtree(f'{path}/Paths')
                # rmtree(f'{path}/Models')