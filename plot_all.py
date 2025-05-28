import os
import platform
from glob import glob

# Define the path to the main script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Determine the operating system
is_windows = platform.system() == "Windows"
# f'{path_to_data}/output/i4-cyber_attack/rl/without_frl/{attack_state}/off-peak/diff_waiting_time_reward_normal_phase_continuity/omega_0.0_cutoff_0_nu_0.5'
# Loop to run the script with different nu values
folder_oi = "./output/i4-cyber_attack/rl/without_frl/attacked/off-peak"

for folder in glob(folder_oi + "/*"):
    if folder.split(os.sep)[-1].startswith("omega_"):
        omega = folder.split(os.sep)[-1].split("_")[1]
        print(f"Processing folder: {folder} with omega: {omega}")
        os.system(f'python plot.py "{folder}" "img/omega/plot_mu_{omega}.png" "attacked" "omega" "{omega}"')