import os
import platform

# Define the path to the main script
script_directory = os.path.dirname(os.path.abspath(__file__))
main_script = "main.py"

# Determine the operating system
is_windows = platform.system() == "Windows"

# Loop to run the script with different nu values
omage = 0.0
for cutoff in [2]:
    omage = omage/100
    for i in [50]:
        nu_value = i / 100.0
        # Construct the command with the script path and nu_value
        # if is_windows:
        #     command = f"START /min python {main_script} --nu {nu_value} > output_{i}.log 2>&1"
        # else:
        for attack_phase in ["True"]:
            print(omage)
            command = f"nohup python3 {main_script} --nu {nu_value} --noise-added '{attack_phase}' --omega {omage} > output_with_reward_continuity_agent_{i}_{omage}_noiseadded_{attack_phase}.log 2>&1 &"
            os.system(command)
            # print(command)
