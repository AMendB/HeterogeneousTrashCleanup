import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('.')
import os

# Read the data
folders = [
    'Evaluation/Results/acoruna_port', 
    'Evaluation/Results/comb_port',
    ]
for folder in folders:
    paths = [
        f'{folder}/DRLIndependentgreedy.4.negativedistance_1_50_2_0/metrics.csv',
        f'{folder}/DRLIndependent.4.negativedistance_1_50_2_0/metrics.csv',
        f'{folder}/Greedy.4.negativedistance_1_50_2_0/metrics.csv',
        f'{folder}/PSO.4.negativedistance_1_50_2_0/metrics.csv',
        f'{folder}/WanderingAgent.4.negativedistance_1_50_2_0/metrics.csv',
        f'{folder}/LawnMower.4.negativedistance_1_50_2_0/metrics.csv',
        ]

    dfs = [pd.read_csv(path) for path in paths]
    max_steps_per_episode = dfs[0]['Step'].max()

    figsize=(9, 7)
    xticks = range(0, max_steps_per_episode+1, 25)
    size_titles = 25
    tick_size = 20

    fig_mse = plt.figure(figsize=figsize)
    ax_mse = fig_mse.add_subplot(111)
    ax_mse.set_title('Mean Squared Error', fontsize=size_titles)
    ax_mse.grid()

    fig_cleaned_percentage = plt.figure(figsize=figsize)
    ax_cleaned_percentage = fig_cleaned_percentage.add_subplot(111)
    ax_cleaned_percentage.set_title('Percentage of Trash Collected', fontsize=size_titles)
    ax_cleaned_percentage.grid()

    for metrics_df in dfs:
        runs = metrics_df['Run'].unique()
        algorithm_name = metrics_df['Algorithm'].unique()[0].split('.')[0]
        if 'DRL' in algorithm_name and 'greedy' in algorithm_name:
            algorithm_name = 'DDDQL + Greedy'
        elif 'DRL' in algorithm_name:
            algorithm_name = 'DDDQL'
        

        # Obtain dataframes #
        numeric_columns = metrics_df.select_dtypes(include=[np.number])

        # Padding each episode with less steps than the max_steps_per_episode with the last value in the episode #
        numeric_columns = numeric_columns.groupby('Run').apply(lambda group: group.set_index('Step').reindex(range(max_steps_per_episode+1), method='ffill').reset_index()).reset_index(drop=True)
        
        # Calculate mean and CI #
        results_mean = numeric_columns.groupby('Step').agg('mean')
        results_confidence_interval = numeric_columns.groupby('Step').agg('sem') * 1.96

        # Extract data to plot or save fig metrics #
        mse = results_mean['MSE'].values.tolist()
        sum_model_changes = results_mean['Sum_model_changes'].values.tolist()
        trash_remaining = results_mean['Trash_remaining'].values.tolist()
        percentage_of_trash_collected = results_mean['Percentage_of_trash_collected'].values.tolist()

        # Plot MSE #
        ax_mse.plot(mse, '-', label=algorithm_name)
        ax_mse.fill_between(results_confidence_interval.index, mse - results_confidence_interval['MSE'], mse + results_confidence_interval['MSE'], alpha=0.2) #, label='Std')
        # ax_mse.legend()
        ax_mse.set_xlim(0, max_steps_per_episode)
        ax_mse.set_xticks(xticks)
        ax_mse.tick_params(axis='x', labelsize=tick_size)
        ax_mse.tick_params(axis='y', labelsize=tick_size)
        # ax_mse.set_yscale('log')
        ax_mse.set_ylabel('MSE', fontsize=size_titles)
        ax_mse.set_xlabel('Step', fontsize=size_titles)
        # ax_mse.annotate(f'{mse[-1]:.2f}', (max_steps_per_episode, mse[-1]), textcoords="offset points", xytext=(-10,0), ha='center')

        # Plot Percentage of trash collected #
        ax_cleaned_percentage.plot(percentage_of_trash_collected, '-', label=algorithm_name)
        ax_cleaned_percentage.fill_between(results_confidence_interval.index, percentage_of_trash_collected - results_confidence_interval['Percentage_of_trash_collected'], percentage_of_trash_collected + results_confidence_interval['Percentage_of_trash_collected'], alpha=0.2) #, label='Std')
        # ax_cleaned_percentage.legend()
        ax_cleaned_percentage.set_xlim(0, max_steps_per_episode)
        ax_cleaned_percentage.set_xticks(xticks)
        ax_cleaned_percentage.tick_params(axis='x', labelsize=tick_size)
        ax_cleaned_percentage.set_yticks(range(0, 101, 20))
        ax_cleaned_percentage.tick_params(axis='y', labelsize=tick_size)
        # ax_cleaned_percentage.set_yscale('log')
        ax_cleaned_percentage.set_ylabel('PTC (%)', fontsize=size_titles)
        ax_cleaned_percentage.set_xlabel('Step', fontsize=size_titles)
        # ax_cleaned_percentage.annotate(f'{percentage_of_trash_collected[-1]:.2f}', (max_steps_per_episode, percentage_of_trash_collected[-1]), textcoords="offset points", xytext=(-10,0), ha='center')

    map_name = folder.split('/')[-1]
    results_folder = '/'.join(folder.split('/')[:-1])
    if map_name == 'acoruna_port':
        map_name = 'scenario_A'
    elif map_name == 'comb_port':
        map_name = 'scenario_B'
    # fig_mse.savefig(f'{results_folder}/MSE_{map_name}.svg')
    fig_mse.savefig(f'{results_folder}/MSE_{map_name}.pdf')
    # fig_cleaned_percentage.savefig(f'{results_folder}/PTC_{map_name}.svg')
    fig_cleaned_percentage.savefig(f'{results_folder}/PTC_{map_name}.pdf')
    plt.show()
