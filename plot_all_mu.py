# -*- coding: utf-8 -*-
"""
Created on Fri Nov 8 11:35:13 2024

@author: akiani
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
mu = 0.11

CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, 'output/i4-cyber_attack/rl/without_frl/attacked/off-peak')

def pad_or_truncate(arr, target_length):
    if len(arr) > target_length:
        return arr[:target_length]
    else:
        return np.pad(arr, (0, target_length - len(arr)), 'constant')
def plot_all_mu(mu):
    alpha = list(range(5))
    DATA_PATH_NEW = os.path.join(CWD, f'output/i4-cyber_attack/rl/neighborhood/3600_values/{mu}/attacked/off-peak')
    reps = 5
    wt = defaultdict(list)
    wt_new = defaultdict(list)  # For the new data
    target_length = 1400  # Define the target length for all arrays

    # Load and process the original data
    for a in alpha:
        for rep in range(reps):
            file_name = 'data_attacked_alpha_{}_run_{}.csv'.format(a, rep)
            file_path = os.path.join(DATA_PATH, file_name)
            if not os.path.isfile(file_path):
                print(file_name)
                continue
            df = pd.read_csv(file_path, header=0)
            df = df.head(target_length)
            wt[a].append(pad_or_truncate(df['system_total_stopped'].values, target_length))
    
    # Load and process the new data from the new folder
    for a in alpha:
        for rep in range(reps):
            file_name = 'data_attacked_alpha_{}_run_{}.csv'.format(a, rep)
            file_path_new = os.path.join(DATA_PATH_NEW, file_name)
            if not os.path.isfile(file_path_new):
                continue
            df_new = pd.read_csv(file_path_new, header=0)
            df_new = df_new.head(target_length)
            wt_new[a].append(pad_or_truncate(df_new['system_total_stopped'].values, target_length))

    # Stack the arrays for easier manipulation
    for a in alpha:
        wt[a] = np.stack(wt[a], axis=0)
        wt_new[a] = np.stack(wt_new[a], axis=0)
    
    # --------------------------------------------------------------------
    df = pd.read_csv('d_2_fixed.csv')
    fixed = df['system_total_stopped'].values
    fixed = pad_or_truncate(fixed, target_length)
    #---------------------------------------------------------------------
    x_ax = np.arange(target_length)
    
    # Plot the original data with solid lines
    _dict = {"x_ax": x_ax, "fixed": fixed}
    for a in alpha:
        mean = wt[a].mean(axis=0)
        _dict[str(a) + "_" + "mean"] = mean
    

    # Plot the new data with dashed lines
    for a in alpha:
        mean_new = wt_new[a].mean(axis=0)
        _dict[str(a) + "_" + "mean_new"] = mean_new

    
    return(_dict)
all_dict = {}
for mu in ["0.11", "0.21", "0.31", "0.41", "0.51", "0.61", "0.71", "0.81", "0.91"]:
    _dict = plot_all_mu(mu)
    all_dict[mu] = _dict

print(all_dict[mu].keys())
        


import matplotlib.pyplot as plt

# Assuming `all_dict` is populated with data as per your script
# Example structure: all_dict[mu]["0_mean"], all_dict[mu]["1_mean"], ... for each alpha

# Define target_length for x-axis
target_length = 550
x_ax = np.arange(target_length)

# Create a plot for each `alpha` across different `mu` values
plt.figure(figsize=(12, 8))
x = []
y = []
for alpha in range(5):
    for mu in list(all_dict.keys()):
        x.append(float(mu))
        y.append(all_dict[mu][f"{alpha}_mean"][-1] - all_dict[mu][f"{alpha}_mean_new"][-1])
    plt.plot(x, y, label="Alpha = {}".format(alpha))
    x = []
    y = []

# Labeling and legend
plt.xlabel('μ (Control Intensity)')
plt.ylabel('Performance Improvement ( actual results - $\mu$ results)')
plt.title('Performance Improvement vs Control Intensity for Different Alpha Values')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), title="Alpha and Mu Values")
plt.grid(True)

# Show the plot
plt.tight_layout()
# plt.show()
plt.savefig('performance_improvement_vs_mu.png', dpi=300)