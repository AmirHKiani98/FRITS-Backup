import random
from copy import deepcopy
import numpy as np
from math import sqrt
from typing import Callable
import traci
import networkx as nx
from src.enviroment.custom_sumorl_env import TrafficSignalCustom
import os
import pandas as pd
def diff_waiting_time_reward_normal(ts:TrafficSignalCustom):
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0

    reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait
    return reward

def diff_waiting_time_reward_normal_phase_continuity(ts:TrafficSignalCustom, reward_fn: Callable):
    reward = reward_fn(ts)

    if ts.get_previous_green_phase() == ts.green_phase:
        reward *= random.uniform(1, 2)
    return reward

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
            distance_matrix[i][j] = distance # type: ignore
            summation += distance
            number += 1
    _mean = summation / number
    
    return (distance_matrix, _mean)

def blend_rewards_neighborhood(rewards:dict, neighbourhood_dict, nu=0.5):
    rewards_copy = deepcopy(rewards)
    for ts, reward in rewards_copy.items():        
        neighbors_rewards = [rewards[neighbour] for neighbour in neighbourhood_dict[ts] if neighbour in rewards]
        if neighbors_rewards:
            rewards_avg = np.mean(neighbors_rewards)
        else:
            rewards_avg = 0.0
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

def blend_rewards(rewards:dict, nu=0.5):
    rewards_avg = np.mean(list(rewards.values()))
    rewards_copy = deepcopy(rewards)
    for ts, reward in rewards_copy.items():
        rewards_copy[ts] = nu*rewards_avg + (1-nu)*reward
    return rewards_copy

def get_graph(net):
    net_prefix = net.split('.')[0] + "_plain"
    os.system(f'netconvert -s {net} --plain-output-prefix {net_prefix}') # TODO if the net doesn't have . in the name, it will work
    # edges to csv
    sumo_path = os.environ.get("SUMO_HOME")
    os.system(f'{sumo_path}/tools/xml/xml2csv.py {net_prefix}.edg.xml')
    data = []
    with open(f"{net_prefix}.edg.csv", "r") as f:
        header = f.readline().strip().split(";")
        
        # lines = f.readlines()

        for line in f:  
            data.append(line.strip().split(";"))
    edges = pd.DataFrame(data, columns=header)
    G = nx.DiGraph()

    # Add edges from CSV
    for _, row in edges.iterrows():
        from_node = str(row['edge_from'])  # Cast to string if needed
        to_node = str(row['edge_to'])
        G.add_edge(from_node, to_node)
    return G

def get_connectivity_network(net, cutoff=2):
    connectivity = {}
    G = get_graph(net)
    for node in G.nodes:
        connectivity[node] = [v for v in list(nx.single_source_shortest_path_length(G, node, cutoff=cutoff).keys()) if v != node] 
    return connectivity

def diff_waiting_time_reward_noised(ts:TrafficSignalCustom, iot_id):
    # if ts.id == iot_id or iot_id == "all":
    #     noise = random.uniform(0.8,1.2)
    # else:
    #     noise = 1

    no_noised_reward = diff_waiting_time_reward_normal(ts)
    reward = 1*(no_noised_reward)
    return reward