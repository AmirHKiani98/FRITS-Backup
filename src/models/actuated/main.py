"""
Main entry point for the application.
"""

import argparse
from src.models.actuated.environmnet.env import ActuateEnv
import os
import traci



def main():
    """
    Main function to parse arguments and run the application.
    """
    print("This is the modular main function for the RL-based traffic signal control system.")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run the main application.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')

    parser.add_argument('--net', type=str, default=BASE_DIR + r"/../../networks/4x4.net.xml")
    parser.add_argument('--route', type=str, default=BASE_DIR + r'/../../routes/4x4c2c1.rou.xml')
    parser.add_argument("--intersection-id", type=str, default="1,5")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--noised-edge", type=str, default="all")
    parser.add_argument("--noise-added", type=bool, default=True)
    parser.add_argument("--min-green", type=int, default=5)
    parser.add_argument("--max-green", type=int, default=50)
    parser.add_argument("--t-crit", type=float, default=10)
    parser.add_argument("--alpha", type=float, default=5, help="Noise level for the state perturbation (0 for no noise).")
    parser.add_argument("--simulation-time", type=int, default=600)


    args = parser.parse_args()
    for episode in range(args.num_episodes):
        env = ActuateEnv(
            route_path=args.route,
            net_path=args.net,
            min_green=args.min_green,
            max_green=args.max_green,
            t_crit=args.t_crit,
            store_results_path="src/models/actuated/output/" + f"data_alpha_{int(args.alpha)}_run_{episode}.csv",
            alpha=args.alpha,
            attacked_intersections=args.intersection_id
        )
        env.start(gui=args.gui)
        for _ in range(args.simulation_time):
            env.step()
        env.store_results()
        env.close()
        

if __name__ == "__main__":
    try:
        main()
    finally:
        if traci.isLoaded():
            traci.close()
        print("Simulation completed and TraCI closed.") 