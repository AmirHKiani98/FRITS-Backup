import subprocess
import os

# With cutoff
for cutoff in [2, 3, 4]:
    for nu in [0.5]:
        command = [
            "python", "-m", "src.main",
            "--cutoff", str(cutoff),
            "--nu", str(nu)
        ]
        print(f"\n➡️ Running: {' '.join(command)}\n")
    
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: Command failed with exit code {e.returncode}")
        os.system(" ".join(command))
    

# With omega
# for omega in [0]:
#     for nu in [0.4, 0.5, 0.6]:
#         command = [
#             "python", "-m" "src.main",
#             "--omega", str(omega),
#             "--nu", str(nu),
#         ]
#         print(f"\n➡️ Running: {' '.join(command)}\n")
    
#         try:
#             subprocess.run(command, check=True)
#         except subprocess.CalledProcessError as e:
#             print(f"❌ Error: Command failed with exit code {e.returncode}")
#         os.system(" ".join(command))
