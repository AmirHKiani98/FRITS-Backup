"""
Main entry point for the application.
"""

import argparse
import os
import traci
from src.enviroment.custom_sumorl_env import CustomSUMORLEnv
from src.enviroment.state_env import ArrivalDepartureState
from src.enviroment.utility import diff_waiting_time_reward_normal_phase_continuity, get_intersections_distance_matrix

from src.rl.dql import DQLAgent

def no_encode(state, ts_id):
    return tuple(state)

def main():
    """
    Main function to parse arguments and run the application.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run the main application.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')

    parser.add_argument('--net', type=str, default=BASE_DIR + r"/networks/4x4.net.xml")
    parser.add_argument('--route', type=str, default=BASE_DIR + r'/routes/4x4c2c1.rou.xml')
    parser.add_argument("--intersection-id", type=str, default="10")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--noised-edge", type=str, default="CR30_LR_8")
    parser.add_argument("--simulation-time", type=int, default=1200)
    parser.add_argument("--run-per-alpha", type=int, default=5)
    parser.add_argument("--delta-time", type=int, default=3)
    parser.add_argument("--nu", type=float, default=0.5)
    parser.add_argument("--distance-threshold", type=int, default=200)
    parser.add_argument("--omega", type=float, default=0.0)
    parser.add_argument("--cutoff", type=int, default=2)


    args = parser.parse_args()


    batch_size = 64
    seed = 7
    num_episodes = args.num_episodes
    

    env = CustomSUMORLEnv(
        net_file=args.net,
        route_file=args.route,
        use_gui=args.gui,
        num_seconds=args.simulation_time,
        min_green=5,
        yellow_time=2,
        delta_time=args.delta_time,
        observation_class=ArrivalDepartureState, # type: ignore
        encode_function=no_encode,
        random_flow=False,
        real_data_type=False,
        percentage_added=0.1,
        reward_fn=diff_waiting_time_reward_normal_phase_continuity
    )

    env.reset()

    distance_matrix, distance_mean = get_intersections_distance_matrix()

    agents = {
        ts: DQLAgent(
            len(set(traci.trafficlight.getControlledLanes(ts))) * 2,
            env.traffic_signals[ts].action_space.n, # type: ignore
            hidden_dim=100,
            seed=seed
        )
        for ts in env.ts_ids
    }

    

    

    