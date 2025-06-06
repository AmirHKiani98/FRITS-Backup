# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:35:13 2024

@author: naftabi
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
path_to_data = sys.argv[1]
path_to_data = path_to_data.strip("\"'")  # Remove any quotes
path_to_data = os.path.normpath(path_to_data)  # Normalize slashes
output_path = sys.argv[2]
output_parent_path = os.path.dirname(output_path)
if not os.path.exists(output_parent_path):
    os.makedirs(output_parent_path)
attack_state = sys.argv[3]
label = sys.argv[4]
CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, f'{path_to_data}')
DATA_PATH_NEW = DATA_PATH.replace("attacked", "no_attack").replace("'", "").replace('"', "")
attribute_oi = "system_total_stopped"
def pad_or_truncate(arr, target_length):
    if len(arr) > target_length:
        return arr[:target_length]
    else:
        return np.pad(arr, (0, target_length - len(arr)), 'constant')
    

if __name__ == '__main__':
    alpha = list(range(5))
    reps = 5
    wt = defaultdict(list)
    wt_new = defaultdict(list)  # For the new data
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
    target_length = 2400  # Define the target length for all arrays

    # Load and process the original data
    for a in alpha:
        for rep in range(reps):
            file_name = 'data_{}_alpha_{}_run_{}.csv'.format("attacked",a, rep)
            file_path = os.path.join(DATA_PATH, file_name)
            if not os.path.isfile(file_path):
                print("File not found:", file_path)
                continue
            df = pd.read_csv(file_path, header=0)
            df = df.iloc[:target_length]
            # print(df.shape, file_name)
            # if df.shape[0] != 18000 and df.shape[0] != 3600:
            #     continue
            df = df.head(target_length)
            wt[a].append(pad_or_truncate(df[attribute_oi].values, target_length))
    
    # Load and process the new data from the new folder
    for a in alpha:
        for rep in range(reps):
            file_name = 'data_{}_alpha_{}_run_{}.csv'.format("no_attack",a, rep)
            file_path_new = os.path.join(DATA_PATH_NEW, file_name)
            if not os.path.isfile(file_path_new):
                print("File not found:", file_path_new)
                continue
            df_new = pd.read_csv(file_path_new, header=0)
            # print(df_new.shape, file_name)
            # if df_new.shape[0] != 18000 and df_new.shape[0] != 600:
            #     continue
            df_new = df_new.head(target_length)
            wt_new[a].append(pad_or_truncate(df_new[attribute_oi].values, target_length))

    # Stack the arrays for easier manipulation
    for a in alpha:
        # print(len(wt[a]), a)
        print(len(wt[a]), len(wt_new[a]), a)
        wt[a] = np.stack(wt[a], axis=0)
        # print(len(wt_new[a]), a)
        wt_new[a] = np.stack(wt_new[a], axis=0)
    
    # --------------------------------------------------------------------
    df = pd.read_csv('4x4_fixed.csv')
    
    fixed = df[attribute_oi].values
    fixed = pad_or_truncate(fixed, target_length)
    #---------------------------------------------------------------------
    x_ax = np.arange(target_length)
    plt.figure(figsize=(18,10))
    
    # Plot the original data with solid lines
    for a in alpha:
        lb = np.percentile(wt[a], 0.25, axis=0)
        ub = np.percentile(wt[a], 99.75, axis=0)
        mean = wt[a].mean(axis=0)
        plt.plot(x_ax, mean, '-', color=colors[a], label=r'$\alpha=$'+'{}'.format(a))
        plt.fill_between(x_ax, lb, ub, color=colors[a], alpha=0.2)

    # Plot the new data with dashed lines
    for a in alpha:
        lb_new = np.percentile(wt_new[a], 0.25, axis=0)
        ub_new = np.percentile(wt_new[a], 99.75, axis=0)
        mean_new = wt_new[a].mean(axis=0)
        plt.plot(x_ax, mean_new, '--', color=colors[a], label=r'$\alpha=$'+'{} | ${}$'.format(a, label))
        plt.fill_between(x_ax, lb_new, ub_new, color=colors[a], alpha=0.2, linestyle='--')

    # Plot the fixed data
    plt.plot(x_ax, fixed, color='k', label='Fixed Time')
    
    # Customize and save the plot
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=20)
    plt.xlabel('Time (s)', fontsize=25)
    plt.ylabel('Total Stopped Vehicles', fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.gca().get_yaxis().get_offset_text().set_size(18)
    plt.savefig(output_path, format='png', dpi=100, bbox_inches='tight')
    print("Figure saved")
    # plt.show()
