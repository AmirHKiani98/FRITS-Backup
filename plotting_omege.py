import os
import platform

# Define the path to the main script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Determine the operating system
is_windows = platform.system() == "Windows"

# Loop to run the script with different nu values
for omage in range(0, 201, 20):
    omage = omage/100
    for attack_state in ["attacked", "no_attack"]:
        command = f"python3 plot.py {omage} {attack_state}"
        os.system(command)
