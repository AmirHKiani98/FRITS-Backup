import os

# Define the path to the main script
script_directory = os.path.dirname(os.path.abspath(__file__))
main_script = os.path.join(script_directory, "main.py")

# Loop to run the script with different nu values
for i in range(1,101, 10):
    nu_value = i / 100.0
    # Construct the command with the script path and nu_value
    command = f"nohup python3 '{main_script}' --nu {nu_value} > output_{i}.log 2>&1 &"
    os.system(command)
