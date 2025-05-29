import os
import platform
import subprocess

# Detect OS
is_windows = platform.system() == "Windows"

# Define base directory and script
script_directory = os.path.dirname(os.path.abspath(__file__))
main_script = os.path.join(script_directory, "main.py")

# Find Python executable in .venv
venv_dir = os.path.join(script_directory, ".venv", "Scripts" if is_windows else "bin")
python_exec = os.path.join(venv_dir, "python.exe" if is_windows else "python3")
os.environ["SUMO_HOME"] = r"F:\Applications\Sumo\bin"
# Confirm the .venv Python exists
if not os.path.exists(python_exec):
    raise FileNotFoundError(f"Python interpreter not found in virtual environment: {python_exec}")

# Run loop
for cutoff in [1, 3]:
    omega = 0
    for i in [50]:
        nu_value = i / 100.0
        for attack_phase in ["True", "False"]:
            log_file = f"output_with_reward_continuity_agent_cutoff_{cutoff}_i_{i}_omega_{omega}_noiseadded_{attack_phase}.log"
            print(f"Launching: nu={nu_value}, omega={omega}, attack={attack_phase}")
            with open(log_file, "w") as f:
                subprocess.Popen(
                    [python_exec, main_script, "--nu", str(nu_value), "--noise-added", attack_phase, "--omega", str(omega), "--cutoff", str(cutoff)],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if is_windows else 0
                )
