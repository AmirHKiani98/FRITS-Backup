"""
Main entry point for the application.
"""

import argparse
import os
import traci
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.enviroment.custom_sumorl_env import CustomSUMORLEnv
from src.enviroment.state_env import ArrivalDepartureState
from src.enviroment.utility import blend_rewards, blend_rewards_neighborhood, diff_waiting_time_reward_normal_phase_continuity, get_connectivity_network, get_intersections_distance_matrix, get_neighbours
from src.rl.dql import DQLAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    connectivity = get_connectivity_network(G, cutoff=args.cutoff)
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
            connectivity
        )
        all_rewards[episode] = episode_rewards

    alphas = list(range(6)) # change value of this list
    for ts, agent in agents.items():
        agent.epsilon = 0.0
        agent.epsilon_end = 0.0
        agent.q_network.eval()


    

def run_episode(env, simulation_time, agents, distance_matrix, distance_mean, omega, cutoff, nu, batch_size, connectivity):
    episode_rewards = []
    state = env.reset()
    for _ in tqdm(range(simulation_time), desc="Processing"):
        
        actions = {ts: agent.act(state[ts]) for ts, agent in agents.items()}
        new_state, reward, _, _ = env.step(action=actions)
        if not isinstance(reward, dict):
            raise ValueError("Reward should be a dictionary with traffic signal IDs as keys.")
        if omega > 0:
            reward = blend_rewards_neighborhood(reward, get_neighbours(distance_mean * omega, distance_matrix), nu)
        elif cutoff > 0:
            reward = blend_rewards_neighborhood(reward, connectivity, nu)
        else:
            reward = blend_rewards(reward, nu)
        for ts, agent in agents.items():
            agent.memory.push(new_state[ts], actions[ts], reward[ts], env.encode(state[ts], ts))
            if len(agent.memory) > batch_size:
                agent.update(batch_size)
        state = new_state # Ask navid about this line
        episode_rewards.append(sum(reward.values()))
    return episode_rewards

def run_alpha(net,
                route,
                gui,
                simulation_time,
                delta_time,
                alpha,
                omega,
                cutoff,
                nu,
                run,
                observation_class,
                agents,
                attack_state
                ):
    env = CustomSUMORLEnv(
                net_file=net,
                route_file=route,
                use_gui=gui,
                num_seconds=simulation_time,
                min_green=5,
                yellow_time=2,
                delta_time=delta_time,
                observation_class=observation_class,
                encode_function=no_encode,
                random_flow=False,
                real_data_type=False,
                percentage_added=0.1,
                reward_fn=diff_waiting_time_reward_normal_phase_continuity
            )
    state = env.reset()
    if not isinstance(state, dict):
        raise ValueError("State should be a dictionary with traffic signal IDs as keys.")
    # env.observation_class.alpha = alpha
    # for ts, agent in agents.items():
    #     agent.memory = deepcopy(memories[ts])

    for step in range(simulation_time): # for how many steps we want to evaluate? 
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
            for ts, agent in agents.items():
                _m = agent.state_dim
                _alpha = np.random.normal(alpha, 1, _m)
                # if alpha == 1 or alpha == 2:
                _alpha = np.abs(_alpha)
                state[ts] = [new_state[ts][i] + _alpha[i] for i in range((_m))] 

        else:
            state = new_state

    # here implement what we want to show as result
    output_folder = os.path.join(
        BASE_DIR,
        "output",
        "i4-cyber_attack",
        "rl",
        "without_frl",
        attack_state,
        "off-peak",
        "diff_waiting_time_reward_normal_phase_continuity",
        f"omega_{omega}_cutoff_{cutoff}_nu_{nu}"
    )
        
    file_name = f"data_{attack_state}_alpha_{alpha}_run_{run}.csv"
    env.custom_save_data(output_folder, file_name=file_name)
    env.delete_cache()