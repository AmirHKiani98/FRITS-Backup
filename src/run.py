import subprocess
import os

# With cutoff
<<<<<<< HEAD
for attacked_intersections in ["10"]:
    for cutoff in [0]:
        for nu in [0.1 ,0.2,0.5,0.6, 0.7, 0.9]:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            output_folder = (
                BASE_DIR + f"/output_modification/4x4/"
                f"{attacked_intersections}_nu_{nu}/"
            )
            output_folder += f"nu_{nu}"
            command = [
                "python", "-m", "src.main",
                "--cutoff", str(cutoff),
                "--nu", str(nu),
                "--intersection-id", str(attacked_intersections),
                "--output-dir", str(output_folder)
            ]
            print(f"\n➡️ Running: {' '.join(command)}\n")
        
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Error: Command failed with exit code {e.returncode}")
            os.system(" ".join(command))
    

# # With omega
# for attacked_intersections in ["1,5", "6,11", "12,16"]:
#     for omega in [0.1]:
#         for nu in [0.5]:
#             BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#             output_folder = (
#                 BASE_DIR + f"/output_modification/4x4/"
#                 f"{attacked_intersections}_cutoff_{cutoff}_nu_{nu}/"
#             )
#             output_folder += f"omega_{omega}"
#             output_folder += f"nu_{nu}"
#             command = [
#                 "python", "-m" "src.main",
#                 "--omega", str(omega),
#                 "--nu", str(nu),
#                 "--output-dir", str(output_folder),
#                 "--intersection-id", str(attacked_intersections),
#             ]
#             print(f"\n➡️ Running: {' '.join(command)}\n")
        
#             try:
#                 subprocess.run(command, check=True)
#             except subprocess.CalledProcessError as e:
#                 print(f"❌ Error: Command failed with exit code {e.returncode}")
#             os.system(" ".join(command))
=======
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
>>>>>>> 5b3b8e77810482cc57dd29426482d8b7081626db
