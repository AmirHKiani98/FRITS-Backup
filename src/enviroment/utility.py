import random
from math import sqrt
from typing import Callable
import traci
from src.enviroment.custom_sumorl_env import TrafficSignalCustom

def diff_waiting_time_reward_normal(ts:TrafficSignalCustom):
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0

    reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait
    return reward

def diff_waiting_time_reward_noised(ts:TrafficSignalCustom, target_intersection_id: list[str]):
    if ts.id in target_intersection_id:
        noise = random.uniform(0.8,1.2)
    else:
        noise = 1

    no_noised_reward = diff_waiting_time_reward_normal(ts)
    reward = noise*(no_noised_reward)
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


def run_simple