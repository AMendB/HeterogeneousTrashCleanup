import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('.')
import os
import json

# Read the data
folder = 'Evaluation/Results'

# Search archive runtime in folder
runtime_json = [f for f in os.listdir(folder) if 'runtime' in f][0]

# Read the data
with open(f"{folder}/{runtime_json}") as f:
    runtimes = json.load(f)

# Rename the keys
if 'DRLIndependentgreedy' in runtimes:
    # runtimes.pop('DRLIndependentgreedy')
    runtimes['DDDQL + Greedy'] = runtimes.pop('DRLIndependentgreedy')
if 'DRLIndependent_Networks_Per_Team' in runtimes:
    # runtimes.pop('DRLIndependent_Networks_Per_Team')
    runtimes['DDDQL'] = runtimes.pop('DRLIndependent_Networks_Per_Team')

# Print mean of the runtimes for each algorithm
for key, value in runtimes.items():
    print(f"{key}: {np.mean(value)*1000:.2f} ms")

# Count cero values
for key, value in runtimes.items():
    print(f"{key}: {len([x for x in value if x == 0])}. Total: {len(value)}")
# Remove cero values
for key, value in runtimes.items():
    runtimes[key] = [x for x in value if x > 0]

# Remove outliers
for key, value in runtimes.items():
    runtimes[key] = [x for x in value if x < 0.05]

# Plot the boxplot 
plt.figure(figsize=(15, 10))
plt.boxplot(runtimes.values(), labels=runtimes.keys())
# plt.show()

# set log scale



# Plot the boxplot with seaborn
# sns.set(style="darkgrid")
# plt.figure(figsize=(15, 10))
# ax = sns.boxplot(data=runtimes, palette='flare', showmeans=True) # hue='Agents combinations' to difference
# ax.legend(title=ax.get_legend().get_title().get_text(), loc='upper left', fontsize=16, title_fontsize=16)
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
plt.ylabel('Time of inference (s)', fontsize=20)
plt.xlabel('')
# plt.title('Boxplots of the MSE for each algorithm and agents combination')
plt.tight_layout()
# plt.savefig(fname=f"{folder}/Runtimes_boxplot.svg")
plt.show()



