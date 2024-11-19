import argparse
import os
import sys

import pandas as pd
from CustomSumoRLEnv import *


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
delta_time = 5
simulation_time = 2000
def get_system_data():
    vehicles = traci.vehicle.getIDList()
    speeds = [traci.vehicle.getSpeed(vehicle) for vehicle in vehicles]
    waiting_times = [traci.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
    return {
        # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
        "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
        "system_total_waiting_time": sum(waiting_times),
        "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
        "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
    } 
def encode(state, ts_id):
    return tuple([_discretize_density(value) for value in state])

def no_encode(state, ts_id):
    return tuple(state)

def _discretize_density(density):
    return min(int(density * 10), 9)
script_directory = os.path.dirname(os.path.abspath(__file__))
# flow_start = [200]
traffic_light_noised_id = "i_cr30_tln"

runs = [1, 20, 100]

attack_state = sys.argv[1]
if attack_state != "attacked":
    attack_state = "no_attack"

learn_last_episode = bool(sys.argv[2])
print(learn_last_episode)
episodes = int(sys.argv[3])

net = r"/Users/kiani014/Documents/Github/FRITS2/learning_controller/maxpressure/net/i4/i4_new.net.xml"
rou = r'/Users/kiani014/Documents/Github/FRITS2/learning_controller/maxpressure/net/i4/rou/d_5_turnCount_pm_peak.rou.xml'
tls_loc = r'/Users/kiani014/Documents/Github/FRITS2/input_efficiency/MP_mndot_data/mp/i4/tls/am_peak_tls.add.xml'

# Reward function:
def diff_waiting_time_reward_noised(ts:TrafficSignal):
    if ts.id == traffic_light_noised_id:
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

if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    decay = 1
    for run in runs:
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
                    if self.ts.id == traffic_light_noised_id:
                        arrival_lane_vehicles += random.randint(0, run) # Adding only positive noise
                        depart_lane_vehicles += random.randint(0, run) # Adding only positive noise
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

        if attack_state == "attacked":
            observation_class = ArrivalDepartureStateAttacked
            reward_fn = diff_waiting_time_reward_noised
        else:
            observation_class = ArrivalDepartureState
            reward_fn = diff_waiting_time_reward_normal
        
        env = CustomSUMORLEnv(
            net_file= net,
            route_file= rou,
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
            # traffic_flow_choices=[1000,1000]
        )
        env.vehicles_flow
        initial_states = env.reset()
        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.traffic_signals[ts].action_space,
                alpha=alpha,
                gamma=gamma,
                exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
            )
            for ts in env.ts_ids
        }
        for episode in range(1, episodes + 1):
            df = pd.DataFrame({})
            print("episode:", episode)
            # flow_start[0] += 1
            
            # env.vehicles_flow = flow_start
            env.percentage_added = episode * 0.1
            if episode != 1:
                initial_states = env.reset()
                # for ts in initial_states.keys():
                #     ql_agents[ts].state = env.encode(initial_states[ts], ts)
            infos = []
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
                

                s, r, done, info = env.step(action=actions)
                if not learn_last_episode:
                    if episode == episodes:
                        print("check")
                        continue
                for agent_id in s.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
            if episode == episodes:
                print('save')
                env.custom_save_data(script_directory + f"/output/i4-cyber_attack/rl/{attack_state}/alpha", file_name=f"data_{attack_state}_run_{run}.csv")
                env.delete_cache()  
        env.close()

    
