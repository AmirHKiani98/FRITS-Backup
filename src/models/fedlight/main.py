"""
Main entry point for the application.
"""

import argparse
from multiprocessing import Pool, cpu_count
import os

import torch
import traci
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models.fedlight.enviroment.custom_sumorl_env import CustomSUMORLEnv
from src.models.fedlight.enviroment.state_env import ConfigurableArrivalDepartureState, create_arrival_departure_state
from src.models.fedlight.agent import Agent as FedLightAgent
from src.models.fedlight.cloud import FedLightCloud
from src.models.fedlight.enviroment.utility import (
    get_pressure
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def no_encode(state, ts_id):
    return tuple(state)

def run_alpha_unpack(args):
    return run_alpha(*args)

def main():
    """
    Main function to parse arguments and run the application.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Run the main application.")

    parser.add_argument('--net', type=str, default=BASE_DIR + r"/../../networks/4x4.net.xml")
    parser.add_argument('--route', type=str, default=BASE_DIR + r'/../../routes/4x4c2c1.rou.xml')
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--noised-edge", type=str, default="6,11")
    parser.add_argument("--noise-added", type=float, default=0.1)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--simulation-time", type=int, default=300)
    parser.add_argument("--run-per-alpha", type=int, default=3)
    parser.add_argument("--delta-time", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--output-dir", type=str, default="")


    args = parser.parse_args()

    num_episodes = args.num_episodes
    reward_fn = lambda ts: get_pressure(ts, args.noised_edge, args.noise_added)
    
    custom_obs_factory = create_arrival_departure_state(alpha=5.0, noise_added=True, attacked_ts=args.noised_edge)
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


    agents = {
        ts: FedLightAgent(
            state_dim=env.traffic_signals[ts].observation_space.shape[0],
            action_dim=int(env.traffic_signals[ts].action_space.n),
            hidden_dim=64,
            actor_lr=0.0001,
            critic_lr=0.0002
        )
        for ts in env.ts_ids
    }
    
    for episode in tqdm(range(num_episodes), desc="Episodes"):
        all_rewards = {}

        episode_rewards = run_episode(
            env,
            args.simulation_time,
            agents,
            args.gamma
        )
        all_rewards[episode] = episode_rewards

    alphas = list(range(6)) # change value of this list
    for _, agent in agents.items():
        # agent.epsilon = 0.0
        # agent.epsilon_end = 0.0
        agent.actor.eval()
    output_folder = args.output_dir if args.output_dir != "" else (
        BASE_DIR + f"/output/i4-fedlight/"
    )
    # for ts, agent in agents.items():
    #     agent.save_policy(idx=ts)
    alpha_tasks = []
    
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
                output_folder,
                args.noised_edge,
                args.noise_added
                
            ]
            )
        
    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(run_alpha_unpack, alpha_tasks), total=len(alpha_tasks), desc="Alpha Tasks"):
            pass
    metadata = {
        "net_file": args.net,
        "route_file": args.route,
        "num_episodes": args.num_episodes,
        "gui": args.gui,
        "simulation_time": args.simulation_time,
        "run_per_alpha": args.run_per_alpha,
        "delta_time": args.delta_time,
        "gamma": args.gamma,
    }
    env.save_metadata(metadata, output_folder, file_name=f"metadata.csv")





def run_episode(env, simulation_time, agents, gamma):
    episode_rewards = []
    state = env.reset()
    trajectory = {ts: [] for ts in agents}
    for _ in tqdm(range(simulation_time), desc="Processing episode"):
        action_probs = {ts: agents[ts].actor(torch.FloatTensor(state[ts]).unsqueeze(0)) for ts in agents}
        actions = {ts: torch.distributions.Categorical(action_probs[ts]).sample().item() for ts in agents}
        new_state, reward, _, _ = env.step(action=actions)
        for ts in agents:
            s = torch.FloatTensor(state[ts])
            a = actions[ts]
            r = reward[ts]
            s_ = torch.FloatTensor(new_state[ts])            
            td_target = r + gamma * agents[ts].critic(s_).item()
            advantage = td_target - agents[ts].critic(s).item()
            trajectory[ts].append((s, a, td_target, advantage))
        state = new_state
        episode_rewards.append(sum(reward.values()))
    for ts in agents:
        agents[ts].compute_gradients(trajectory[ts])
    cloud = FedLightCloud()

    for ts in agents:
        cloud.collect(ts, agents[ts].get_gradients())

    avg_grads = cloud.average_and_dispatch()

    for ts in agents:
        agents[ts].apply_gradients(avg_grads)
        agents[ts].step()
    
    return episode_rewards

def run_alpha(net,
               route,
               gui,
               simulation_time,
               delta_time,
               alpha,
               run,
               observation_class,
               agents,
               output_folder,
               noised_edge,
               noise_added
               ):
    reward_fn = lambda ts: get_pressure(ts, noised_edge, noise_added)
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
            for ts, agent in agents.items():
                _m = agent.state_dim
                _alpha = np.random.normal(alpha, 1, _m)
                # if alpha == 1 or alpha == 2:
                _alpha = np.abs(_alpha)
                state[ts] = [new_state[ts][i] + _alpha[i] for i in range((_m))] 

        else:
            state = new_state

    # here implement what we want to show as result
        
    file_name = f"data_fedlight_alpha_{alpha}_run_{run}.csv"
    env.custom_save_data(output_folder, file_name=file_name)
    env.delete_cache()

if __name__ == "__main__":
    try:
        main()
    finally:
        if traci.isLoaded():
            traci.close()
        print("Simulation completed and TraCI closed.")