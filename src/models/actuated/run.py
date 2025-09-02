import os


for alpha in [5]:
    alpha = int(alpha)
    os.system(f"python -m src.models.actuated.main --alpha {alpha} --num-episodes 5 --noise-added True --noised-edge '1,5'")