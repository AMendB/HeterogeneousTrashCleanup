import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_multiple_losses(csv_folder_path, team_id):

    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]

    # Filter by team_id
    if 'loss' in csv_folder_path.lower():
        csv_files = [f for f in csv_files if f'log{team_id}' in f]


    if not csv_files:
        print("No CSV files found in the folder")
        return

    plt.figure(figsize=(8, 6))

    for csv_file in csv_files:
        # Read the CSV file
        df = pd.read_csv(os.path.join(csv_folder_path, csv_file))

        # Training names:
        if 'greedy' in csv_file.lower():
            training_name = 'DDDQL + Greedy'
        elif 'pso' in csv_file.lower():
            training_name = 'DDDQL + PSO'
        else:
            training_name = 'DDDQL'

        # Plot without smoothing
        # plt.plot(df['Step'], df['Value'], label=f'{training_name}')

        # Plot with smoothing
        plt.plot(df['Step'], df['Value'].rolling(window=4).mean(), label=f'{training_name}')

    if team_id == 0:
        team = 'Scouting Team'
    else:
        team = 'Cleaning Team'

    if 'acoruna' in csv_folder_path.lower():
        scenario_name = 'Scenario A'
    elif 'comb' in csv_folder_path.lower():
        scenario_name = 'Scenario B'
    elif 'challenging' in csv_folder_path.lower():
        scenario_name = 'Scenario C'

    if 'loss' in csv_folder_path.lower():
        plt.title(f'Loss during training {scenario_name} - {team}', fontsize=17)
        plt.ylabel('Loss', fontsize=15)
        metric = 'loss'

        # Limit y axis
        plt.ylim(0, 60)
    elif 'mse' in csv_folder_path.lower():
        plt.title(f'Avg. MSE (100 test episodes) during training {scenario_name}', fontsize=16)
        plt.ylabel('Mean Squared Error', fontsize=15)
        metric = 'mse'
    elif 'ptc' in csv_folder_path.lower():
        plt.title(f'Avg. PTC (100 test episodes) during training {scenario_name}', fontsize=16)
        plt.ylabel('Percentage of Trash Collected', fontsize=15)
        metric = 'ptc'
    plt.xlabel('Step', fontsize=15)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    plt.grid(True)
    plt.legend()
    
    if metric == 'loss':
        plt.savefig(f'{csv_folder_path}/{metric}_plot_{team_id}.png')
    else:
        plt.savefig(f'{csv_folder_path}/{metric}_plot.png')
    plt.show()

if __name__ == "__main__":
    plot_multiple_losses('Training/T/TensorboardChallenging/MSE', team_id=0)