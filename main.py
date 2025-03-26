# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:46:51 2024

@authors: naftabi, akiani
"""
import os
import argparse
import pandas as pd
from copy import deepcopy
from math import sqrt
import pandas as pd
import networkx as nx

from dql import DQLAgent

import xml.etree.ElementTree as ET

import numpy as np
from collections import defaultdict

from CustomSumoRLEnv import *

from tqdm import tqdm



parser = argparse.ArgumentParser()
# Command to run the file: python main.py --noise-added "True"
script_directory = os.path.dirname(os.path.abspath(__file__))
parser.add_argument('--net', type=str, default=script_directory + r"/net/4x4.net.xml")
parser.add_argument('--route', type=str, default=script_directory + r'/rou/4x4c2c1.rou.xml')
parser.add_argument('--noise-added', type=str, default="True")
parser.add_argument("--intersection-id", type=str, default="10")
parser.add_argument("--num-episodes", type=int, default=5)
parser.add_argument("--gui", type=bool, default=False)
parser.add_argument("--noised-edge", type=str, default="CR30_LR_8")
parser.add_argument("--simulation-time", type=int, default=200)
parser.add_argument("--run-per-alpha", type=int, default=5)
parser.add_argument("--delta-time", type=int, default=3)
parser.add_argument("--nu", type=float, default=0.5)
parser.add_argument("--distance-threshold", type=int, default=200)
parser.add_argument("--omega", type=float, default=0.0)
parser.add_argument("--cutoff", type=int, default=0)

args = parser.parse_args()
print(args.noise_added, "noise")
if args.noise_added.lower() == "false":
    args.noise_added = False
else:
    args.noise_added = True
delta_time = args.delta_time
nu = args.nu
simulation_time_value = int(float(args.simulation_time) * float(args.delta_time))
if args.noise_added:
    attack_state = "attacked"
else:
    attack_state = "no_attack"

# network to edges
net_prefix = args.net.split('.')[0] + "_plain"
os.system(f'netconvert -s {args.net} --plain-output-prefix {net_prefix}') # TODO if the net doesn't have . in the name, it will work
# edges to csv
sumo_path = os.environ.get("SUMO_HOME")
os.system(f'{sumo_path}/tools/xml/xml2csv.py {net_prefix}.edg.xml')

data = []
with open(f"{net_prefix}.edg.csv", "r") as f:
    header = f.readline().strip().split(";")
    
    # lines = f.readlines()

    for line in f:  
        # print(f"{i+1}: {line.strip()}")
        data.append(line.strip().split(";"))
edges = pd.DataFrame(data, columns=header)

# csv to graph
# Create a directed graph
G = nx.DiGraph()

# Add edges from CSV
for _, row in edges.iterrows():
    from_node = str(row['edge_from'])  # Cast to string if needed
    to_node = str(row['edge_to'])
    G.add_edge(from_node, to_node)
    
def get_connectivity_network(G, cutoff=2):
    connectivity = {}
    for node in G.nodes:
        connectivity[node] = [v for v in list(nx.single_source_shortest_path_length(G, node, cutoff=cutoff).keys()) if v != node] 
    return connectivity

connectivity = get_connectivity_network(G, cutoff=args.cutoff)
print("attack state:", attack_state)


def get_intersections_positions():
    junction_ids = traci.junction.getIDList()
    junction_positions = {jid: traci.junction.getPosition(jid) for jid in junction_ids}
    return junction_positions

def get_intersections_distance_matrix():
    junction_positions = get_intersections_positions()
    distance_matrix = {key: {nested_key: 0 for nested_key in junction_positions.keys()} for key in junction_positions.keys()}
    summation = 0
    number = 0
    for i in junction_positions.keys():
        for j in junction_positions.keys():
            distance = sqrt((junction_positions[i][0] - junction_positions[j][0])**2 + (junction_positions[i][1] - junction_positions[j][1])**2)
            distance_matrix[i][j] = distance
            summation += distance
            number += 1
    _mean = summation / number
    
    return (distance_matrix, _mean)

def no_encode(state, ts_id):
    return tuple(state)



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

    def get_lane_vehicles(self, lane_id, distance_threshold=200):
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
    

class ArrivalDepartureStateAttacked(ArrivalDepartureState):
    alpha = 0
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)
        
        

    def __call__(self):
        """Subclasses must override this method."""
        arrival = []
        departure = []
        for index, arrival_lane in enumerate(self.incoming_lanes):
            arrival_lane_vehicles = self.get_lane_vehicles(lane_id=arrival_lane)
            depart_lane_vehicles = 0
            for departure_lane in self.outgoing_lanes[index]:
                depart_lane_vehicles += self.get_lane_vehicles(lane_id=departure_lane)
            if self.ts.id == args.intersection_id: #and traci.lane.getEdgeID(arrival_lane) == args.noised_edge: #TODO check for noised_edge later
                arrival_lane_vehicles += random.randint(0, ArrivalDepartureStateAttacked.alpha) # Adding only positive noise
                depart_lane_vehicles += random.randint(0, ArrivalDepartureStateAttacked.alpha) # Adding only positive noise
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

    def get_lane_vehicles(self, lane_id, distance_threshold=100):
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
            high=np.inf,
            shape=self.state_dim
        )
# Reward function:
def diff_waiting_time_reward_noised(ts:TrafficSignal):
    if ts.id == args.intersection_id:
        noise = random.uniform(0.8,1.2)
    else:
        noise = 1

    no_noised_reward = diff_waiting_time_reward_normal(ts)
    reward = noise*(no_noised_reward)
    return reward

def diff_waiting_time_reward_normal(ts:TrafficSignal):
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait
    return reward

def blend_rewards(rewards:dict, nu=0.5):
    rewards_avg = np.mean(list(rewards.values()))
    rewards_copy = deepcopy(rewards)
    for ts, reward in rewards_copy.items():
        rewards_copy[ts] = nu*rewards_avg + (1-nu)*reward
    return rewards_copy

def get_neighbours(distance_threshold, distance_matrix):
    neighbours = {}
    for ts in distance_matrix.keys():
        neighbours[ts] = []
        for ts2 in distance_matrix[ts].keys():
            if ts != ts2 and distance_matrix[ts][ts2] <= distance_threshold:
                neighbours[ts].append(ts2)
    return neighbours


def blend_rewards_neighborhood(rewards:dict, neighbourhood_dict, nu=0.5):
    rewards_copy = deepcopy(rewards)
    for ts, reward in rewards_copy.items():
        rewards_avg = np.mean([rewards[neighbour] for neighbour in neighbourhood_dict[ts]])
        rewards_copy[ts] = nu*rewards_avg + (1-nu)*reward
    return rewards_copy

if args.noise_added:
    reward_fn = diff_waiting_time_reward_noised
else:
    reward_fn = diff_waiting_time_reward_normal

observation_class = ArrivalDepartureState
if True:
    
    batch_size = 64
    seed = 7
    num_episodes = args.num_episodes
    
    # env parameters
    # state_dim = 10
    # action_dim = 10

    env = CustomSUMORLEnv(
        net_file=args.net,
        route_file=args.route,
        use_gui=args.gui,
        num_seconds=args.simulation_time,
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
    # Get neighbors for each traffic signal based on distance threshold
    distance_matrix, distance_mean = get_intersections_distance_matrix()
    
    agents = {ts: DQLAgent(len(set(traci.trafficlight.getControlledLanes(ts)))*2, env.traffic_signals[ts].action_space.n, hidden_dim=100, seed=seed) for ts in env.ts_ids}
    
    total_rewards = []  # Store total rewards for each episode
    all_rewards = defaultdict(list)  # Store all rewards for all steps
    for episode in range(num_episodes):
        state = env.reset()
        print("Episode: ", episode)
        for step in tqdm(range(args.simulation_time), desc="Processing"):
            
            actions = {ts: agent.act(state[ts]) for ts, agent in agents.items()}
            
            # Next step
            new_state, reward, _, _ = env.step(action=actions)
            if args.omega > 0:
                reward = blend_rewards_neighborhood(reward, get_neighbours(distance_mean * args.omega, distance_matrix), nu)
            elif args.cutoff > 0:
                reward = blend_rewards_neighborhood(reward, connectivity, nu)
            else:
                reward = blend_rewards(reward, nu)
            for ts, agent in agents.items():
                agent.memory.push(new_state[ts], actions[ts], reward[ts], env.encode(state[ts], ts))
                if len(agent.memory) > batch_size:
                    agent.update(batch_size)
            state = new_state # Ask navid about this line
            all_rewards[episode].append(sum(reward.values()))
        print('Episode: {}, Total Reward: {}'.format(episode + 1, sum(all_rewards[episode])))


    import matplotlib.pyplot as plt

    
    alphas = list(range(6)) # change value of this list
    for ts, agent in agents.items():
        agent.epsilon = 0.0
        agent.epsilon_end = 0.0
        agent.q_network.eval()
    
    for alpha in alphas:
        for run in range(args.run_per_alpha):
            print("alpha", alpha, "run", run)
            env = CustomSUMORLEnv(
                net_file=args.net,
                route_file=args.route,
                use_gui=args.gui,
                num_seconds=args.simulation_time,
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
            # env.observation_class.alpha = alpha
            # for ts, agent in agents.items():
            #     agent.memory = deepcopy(memories[ts])

            for step in range(args.simulation_time): # for how many steps we want to evaluate? 
                actions = {ts: agent.act(state[ts]) for ts, agent in agents.items()}
                try:
                    new_state, reward, _, _ = env.step(action=actions)
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
            output_folder = script_directory + f"/output/i4-cyber_attack/rl/without_frl/{attack_state}/off-peak/omega_{args.omega}_cutoff_{args.cutoff}_nu_{args.nu}/"
                
            env.custom_save_data(output_folder, file_name=f"data_{attack_state}_alpha_{alpha}_run_{run}.csv")
            env.delete_cache()
            # env.close()
    metadata = {
        "net_file": args.net,
        "route_file": args.route,
        "noise_added": args.noise_added,
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
    