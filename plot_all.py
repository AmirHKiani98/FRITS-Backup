import os
import platform
from glob import glob
# Define the path to the main script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Determine the operating system
is_windows = platform.system() == "Windows"
# f'{path_to_data}/output/i4-cyber_attack/rl/without_frl/{attack_state}/off-peak/diff_waiting_time_reward_normal_phase_continuity/omega_0.0_cutoff_0_nu_0.5'
# Loop to run the script with different nu values
folder_oi = "/Users/cavelab/Documents/Github/FRITS-Backup/output/i4-cyber_attack/rl/without_frl/attacked/off-peak"
for folder in glob(folder_oi + "/*"):
    if folder.split("/")[-1].startswith("omega_") and "nu" not in folder.split("/")[-1]:
        omega = folder.split("/")[-1].split("_")[1]
        os.system(f"python3 plot.py '{folder}' 'img/omega/plot_omega_{omega}.png' 'attacked' 'omega' '{omega}'")
