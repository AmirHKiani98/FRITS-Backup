# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:46:51 2024

@author: naftabi
"""

import sys
import argparse
import pandas as pd
parser = argparse.ArgumentParser()
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
parser.add_argument('--net', type=str, default=script_directory +  r"/net/i4_new.net.xml")
parser.add_argument('--route', type=str, default=script_directory + r'/rou/d_5_turnCount_pm_peak.rou.xml')
parser.add_argument('--noise-added', type=bool, default=False)
parser.add_argument("--intersection-id", type=str, default="i_cr30_tln")
parser.add_argument("--last-episode-not-learn", type=bool, default=False)
parser.add_argument("--alpha", type=int, default=0)
parser.add_argument("--num-episodes", type=int, default=5)
parser.add_argument("--max-steps", type=int, default=600)

args = parser.parse_args()

delta_time = 5
simulation_time = 2000

if args.noise_added:
    attack_state = "attacked"
else:
    attack_state = "no_attack"

import numpy as np
from collections import defaultdict

from CustomSumoRLEnv import *

def no_encode(state, ts_id):
    return tuple(state)

from dql import DQLAgent

class ArrivalDepartureState(ObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

        self.arrival_lanes = []
        self.departure_lanes = []
        self.incoming_lanes = list(set(traci.trafficlight.getControlledLanes(ts.id)))
        # Get the list of outgoing lanes
        self.outgoing_lanes = []
        for lane in self.incoming_lanes:
            successors = traci.lane.getLinks(lane)
            temp_list = []
            for successor in successors:
                temp_list.append(successor[0])
            self.outgoing_lanes.append(temp_list)
        # queue = self.ts.get_lanes_queue()
        self.state_dim = (len(self.incoming_lanes)*2, )
        
        self.intersection_pos = traci.junction.getPosition(self.ts.id)
        
            

    def __call__(self):
        """Subclasses must override this method."""
        arrival = []
        departure = []
        for index, arrival_lane in enumerate(self.incoming_lanes):
            arrival_lane_vehicles = self.get_lane_vehicles(lane_id=arrival_lane)
            depart_lane_vehicles = 0
            for departure_lane in self.outgoing_lanes[index]:
                depart_lane_vehicles += self.get_lane_vehicles(lane_id=departure_lane)
            arrival.append(arrival_lane_vehicles)
            departure.append(depart_lane_vehicles)
        return arrival + departure
    
    def generate_random_integers(self, start, end, length):
        if start > end:
            start, end = end, start  # Swap start and end if start is greater than end

        random_integers = [random.randint(start, end) for _ in range(length)]
        return random_integers

    def get_lane_vehicles(self, lane_id, distance_threshold=50):
        vehicle_ids_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
        summation = 0
        for vehicle_id in vehicle_ids_on_lane:
            vehicle_position = traci.vehicle.getPosition(vehicle_id)
            distance = traci.simulation.getDistance2D(vehicle_position[0], vehicle_position[1], self.intersection_pos[0], self.intersection_pos[1])
            if distance < distance_threshold:
                summation += 1
        return summation

    def observation_space(self):
        """Subclasses must override this method."""
        return spaces.Box(
            low=0,
            high=500,
            shape=self.state_dim
        )

class ArrivalDepartureStateAttacked(ObservationFunction):

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

        self.arrival_lanes = []
        self.departure_lanes = []
        self.incoming_lanes = list(set(traci.trafficlight.getControlledLanes(ts.id)))
        # Get the list of outgoing lanes
        self.outgoing_lanes = []
        for lane in self.incoming_lanes:
            successors = traci.lane.getLinks(lane)
            temp_list = []
            for successor in successors:
                temp_list.append(successor[0])
            self.outgoing_lanes.append(temp_list)
        # queue = self.ts.get_lanes_queue()
        self.state_dim = (len(self.incoming_lanes)*2, )
        
        self.intersection_pos = traci.junction.getPosition(self.ts.id)
        

    def __call__(self):
        """Subclasses must override this method."""
        arrival = []
        departure = []
        for index, arrival_lane in enumerate(self.incoming_lanes):
            arrival_lane_vehicles = self.get_lane_vehicles(lane_id=arrival_lane)
            depart_lane_vehicles = 0
            for departure_lane in self.outgoing_lanes[index]:
                depart_lane_vehicles += self.get_lane_vehicles(lane_id=departure_lane)
            if self.ts.id == args.intersection_id:
                arrival_lane_vehicles += random.randint(0, args.alpha) # Adding only positive noise
                depart_lane_vehicles += random.randint(0, args.alpha) # Adding only positive noise
            arrival_lane_vehicles = max(arrival_lane_vehicles, 0)
            depart_lane_vehicles = max(depart_lane_vehicles, 0)
             
            arrival.append(arrival_lane_vehicles)
            departure.append(depart_lane_vehicles)
        
            
        return arrival + departure
    
    def generate_random_integers(self, start, end, length):
        if start > end:
            start, end = end, start  # Swap start and end if start is greater than end

        random_integers = [random.randint(start, end) for _ in range(length)]
        return random_integers

    def get_lane_vehicles(self, lane_id, distance_threshold=50):
        vehicle_ids_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
        summation = 0
        for vehicle_id in vehicle_ids_on_lane:
            vehicle_position = traci.vehicle.getPosition(vehicle_id)
            distance = traci.simulation.getDistance2D(vehicle_position[0], vehicle_position[1], self.intersection_pos[0], self.intersection_pos[1])
            if distance < distance_threshold:
                summation += 1
        return summation

    def observation_space(self):
        """Subclasses must override this method."""
        return spaces.Box(
            low=0,
            high=500,
            shape=self.state_dim
        )
# Reward function:
def diff_waiting_time_reward_noised(ts:TrafficSignal):
    if ts.id == args.intersection_id:
        noise = random.uniform(0.8,1.2)
    else:
        noise = 1

    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    reward = noise*(ts.last_measure - ts_wait)
    ts.last_measure = ts_wait
    return reward

def diff_waiting_time_reward_normal(ts:TrafficSignal):
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait
    return reward

if args.noise_added:
    reward_fn = diff_waiting_time_reward_noised
else:
    reward_fn = diff_waiting_time_reward_normal

observation_class = ArrivalDepartureState
if __name__ == '__main__':
    
    max_steps = args.max_steps
    batch_size = 64
    seed = 7
    num_episodes = args.num_episodes
    
    # env parameters
    # state_dim = 10
    # action_dim = 10

    env = CustomSUMORLEnv(
        net_file=args.net,
        route_file=args.route,
        use_gui=False,
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
    env.reset()
    agents = {ts: DQLAgent(len(set(traci.trafficlight.getControlledLanes(ts)))*2, env.traffic_signals[ts].action_space.n, hidden_dim=64, seed=seed) for ts in env.ts_ids}
    
    
    total_rewards = []  # Store total rewards for each episode
    all_rewards = defaultdict(list)  # Store all rewards for all steps
    average_reward_per_episode = 0
    epsilon_done = False
    for episode in range(num_episodes):

        # reset environment
        state = env.reset()
        for step in range(max_steps):
            actions = {ts: agent.act(state[ts]) for ts, agent in agents.items()}
            print(actions)
            # Next step
            new_state, reward, _, _ = env.step(action=actions)
            state = new_state
            all_rewards[episode].append(reward)

            if args.last_episode_not_learn:
                if episode == num_episodes - 1:
                    if not epsilon_done:
                        
                        for ts, agent in agents.items():
                            agent.epsilon_end = 0
                            agent.epsilon = 0
                        
                        if args.noise_added:
                            env.observation_class = ArrivalDepartureStateAttacked
                        epsilon_done = True
                    
                    continue
            
            for ts, agent in agents.items():
                agent.memory.push(new_state[ts], actions[ts], -1*reward[ts], env.encode(state[ts], ts))
                if len(agent.memory) > batch_size:
                    agent.update(batch_size)
            
        total_rewards.append(pd.DataFrame(all_rewards[episode]).sum())
        
        average_reward_per_episode = ((average_reward_per_episode * (episode)) + total_rewards[episode].mean())/(episode+1)
        sys.stdout.write(
            "Episode: {:<3}, Total Reward: {:<15}, Avg. Reward: {:<15}  \n".format(episode + 1, 
                                                                            np.round(total_rewards[episode].mean(), 2),
                                                                            np.round(average_reward_per_episode, 2)))
        if episode == num_episodes-1:
            print("save")
            env.custom_save_data(script_directory + f"/output/i4-cyber_attack/rl/{attack_state}/alpha", file_name=f"data_{attack_state}_run_{args.alpha}.csv")
        env.delete_cache()