"""
Main entry point for the application.
"""

import argparse
from multiprocessing import Pool, cpu_count
import os
import traci
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.enviroment.custom_sumorl_env import CustomSUMORLEnv
from src.enviroment.state_env import create_arrival_departure_state
from src.enviroment.utility import blend_rewards, blend_rewards_neighborhood, diff_waiting_time_reward_noised, diff_waiting_time_reward_normal, diff_waiting_time_reward_normal_phase_continuity, get_connectivity_network, get_intersections_distance_matrix, get_neighbours
from src.rl.dql import DQLAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def no_encode(state, ts_id):
    return tuple(state)

def run_alpha_unpack(args):
    return run_alpha(*args)

class NoisedRewardFunction:
    """A pickleable reward function class for noised scenarios."""
    def __init__(self, noised_edge):
        self.noised_edge = noised_edge
    
    def __call__(self, ts):
        return diff_waiting_time_reward_noised(ts, self.noised_edge)

def main():
    """
    Main function to parse arguments and run the application.
    """
    print("This is the modular main function for the RL-based traffic signal control system.")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run the main application.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')

    parser.add_argument('--net', type=str, default=BASE_DIR + r"/networks/4x4.net.xml")
    parser.add_argument('--route', type=str, default=BASE_DIR + r'/routes/4x4c2c1.rou.xml')
    parser.add_argument("--intersection-id", type=str, default="1,5")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--gui", type=bool, default=False)
    parser.add_argument("--noised-edge", type=str, default="all")
    parser.add_argument("--noise-added", type=bool, default=True)
    parser.add_argument("--alpha", type=float, default=5, help="Noise level for the state perturbation (0 for no noise).")
    parser.add_argument("--simulation-time", type=int, default=300)
    parser.add_argument("--run-per-alpha", type=int, default=3)
    parser.add_argument("--delta-time", type=int, default=3)
    parser.add_argument("--nu", type=float, default=0.5)
    parser.add_argument("--distance-threshold", type=int, default=30)
    parser.add_argument("--omega", type=float, default=0.0)
    parser.add_argument("--cutoff", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="")
    ## Run the next two simulations first
    # TODO: Don't the next TODO first. Do this TODO first. This TODO is to run different simulations for different mu for only one intersection (10)
    # TODO: Try running the simulation for when we only have mu but for two intersections (1,5)





    # TODO: Make alpha proportional to the number of vehicle on the approach.
    # TODO: Our model is not performing well without attack. Let's not use cutoff. (Reverse back)
    # TODO: Another scenarios: Weight the reward by the distance too
    # TODO: The issues might be because of the design of the attacks.
    # TODO: Try changing the delta_time

    args = parser.parse_args()

    batch_size = 64
    seed = 7
    num_episodes = args.num_episodes
    if args.noise_added:
        reward_fn = NoisedRewardFunction(args.noised_edge)
        attack_state = "attacked"
    else:
        reward_fn = diff_waiting_time_reward_normal
        attack_state = "no_attack"
    # Create a pickleable observation factory for multiprocessing
    custom_obs_factory = create_arrival_departure_state(alpha=0, noise_added=True, attacked_ts=args.noised_edge)
    env = CustomSUMORLEnv(
        net_file=args.net,
        route_file=args.route,
        use_gui=args.gui,
        num_seconds=args.simulation_time,
        min_green=5,
        yellow_time=2,
        delta_time=args.delta_time,
        observation_class=custom_obs_factory, # type: ignore
        encode_function=no_encode,
        random_flow=False,
        real_data_type=False,
        percentage_added=0.1,
        reward_fn=reward_fn
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
    connectivity = get_connectivity_network(args.net, cutoff=args.cutoff)
    for episode in tqdm(range(num_episodes), desc="Episodes"):
        all_rewards = {}

        episode_rewards = run_episode(
            env,
            args.simulation_time,
            agents,
            distance_matrix,
            distance_mean,
            args.omega,
            args.cutoff,
            args.nu,
            batch_size,
            connectivity,
            reward_fn=reward_fn
        )
        all_rewards[episode] = episode_rewards

    alphas = list(range(6)) # change value of this list
    for _, agent in agents.items():
        agent.epsilon = 0.0
        agent.epsilon_end = 0.0
        agent.q_network.eval()
    output_folder = args.output_dir
    
    alpha_tasks = []
    
    if not attack_state == "attacked":
        alphas = [0] # If no attack, only run alpha = 0
    for alpha in alphas:
        for run in range(args.run_per_alpha):
            alpha_tasks.append([
                args.net,
                args.route,
                args.gui,
                args.simulation_time,
                args.delta_time,
                alpha,
                run,
                custom_obs_factory, # type: ignore
                agents,
                attack_state,
                output_folder,
                reward_fn
            ]
            )
        
    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(run_alpha_unpack, alpha_tasks), total=len(alpha_tasks), desc="Alpha Tasks"):
            pass
    if traci.isLoaded():
        traci.close()    
    metadata = {
        "net_file": args.net,
        "route_file": args.route,
        "noise_added":  args.noise_added,
        "intersection_id": args.intersection_id,
        "num_episodes": args.num_episodes,
        "gui": args.gui,
        "noised_edge": args.noised_edge,
        "simulation_time": args.simulation_time,
        "run_per_alpha": args.run_per_alpha,
        "delta_time": args.delta_time,
        "nu": args.nu,
        "distance_threshold": args.distance_threshold,
        "omega": args.omega,
        "distance_mean": distance_mean
    }
    env.save_metadata(metadata, output_folder, file_name=f"metadata_{attack_state}.csv")

    

def run_episode(env, simulation_time, agents, distance_matrix, distance_mean, 
                omega, cutoff, nu, batch_size, connectivity, reward_fn):
    episode_rewards = []
    state = env.reset()
    for _ in tqdm(range(simulation_time), desc=f"Run the episode", total=simulation_time):
        actions = {ts: agent.act(state[ts]) for ts, agent in agents.items()}
        new_state, reward, _, _ = env.step(action=actions)
        reward = {ts_id: reward_fn(ts) for ts_id, ts in env.traffic_signals.items()}
        if not isinstance(reward, dict):
            raise ValueError("Reward should be a dictionary with traffic signal IDs as keys.")
        
        if omega > 0:
            reward = blend_rewards_neighborhood(
                reward, get_neighbours(distance_mean * omega, distance_matrix), nu
            )
        elif cutoff > 0:
            reward = blend_rewards_neighborhood(reward, connectivity, nu)
        else:
            reward = blend_rewards(reward, nu)
        
        for ts, agent in agents.items():
            agent.memory.push(
                env.encode(state[ts], ts), actions[ts], reward[ts], new_state[ts], False
            )
            if len(agent.memory) > batch_size:
                agent.update(batch_size)
        
        state = new_state
        episode_rewards.append(sum(reward.values()))
    return episode_rewards

def run_alpha(net,
                route,
                gui,
                simulation_time,
                delta_time,
                alpha,
                run,
                observation_factory,
                agents,
                attack_state,
                output_folder,
                reward_fn
                ):
    env = CustomSUMORLEnv(
                net_file=net,
                route_file=route,
                use_gui=gui,
                num_seconds=simulation_time,
                min_green=5,
                yellow_time=2,
                delta_time=delta_time,
                observation_class=observation_factory,
                encode_function=no_encode,
                random_flow=False,
                real_data_type=False,
                percentage_added=0.1,
                reward_fn=reward_fn
            )
    state = env.reset()
    if not isinstance(state, dict):
        raise ValueError("State should be a dictionary with traffic signal IDs as keys.")

    for _ in range(simulation_time): # for how many steps we want to evaluate? 
        actions = {ts: agent.act(state[ts]) for ts, agent in agents.items()}
        try:
            result = env.step(action=actions)
            if isinstance(result, tuple) and len(result) == 5:
                new_state, _, _, _, _ = result
            else:
                new_state, _, _, _ = result
        except:
            break
        
        if alpha > 0:
            # Apply noise to the state for the agents, but keep the actual environment state
            noisy_state = {}
            for ts, agent in agents.items():
                _m = agent.state_dim
                _alpha = np.random.randint(0, alpha, _m).astype(int)
                _alpha = np.abs(_alpha)
                noisy_state[ts] = [new_state[ts][i] + _alpha[i] for i in range(_m)]
            state = noisy_state
        else:
            state = new_state

    # here implement what we want to show as result
        
    file_name = f"data_{attack_state}_alpha_{alpha}_run_{run}.csv"
    env.custom_save_data(output_folder, file_name=file_name)
    env.delete_cache()

if __name__ == "__main__":
    try:
        main()
    finally:
        if traci.isLoaded():
            traci.close()
        print("Simulation completed and TraCI closed.")