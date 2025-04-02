
from typing import Callable, Optional, Tuple, Union
from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import DefaultObservationFunction, ObservationFunction
import numpy as np
from collections import defaultdict
from sumo_rl.environment.traffic_signal import TrafficSignal
import traci
import numpy as np
import random
from gymnasium import spaces
import string
import pandas as pd
import os
import re
import json
from pathlib import Path

class CustomObservation(ObservationFunction):

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
        no_phases = len(traci.trafficlight.getAllProgramLogics(self.ts.id)[0].phases)//2
        self.state_dim = (len(self.incoming_lanes)*2+no_phases, )

    def __call__(self):
        """Subclasses must override this method."""
        incoming_lanes_state = []
        outgoing_lanes_state = []
        
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        for index, arrival_lane in enumerate(self.incoming_lanes):
            incoming_lanes_state.append(traci.lane.getLastStepVehicleNumber(arrival_lane))
            depart_number = 0
            for departure_lane in self.outgoing_lanes[index]:
                depart_number += traci.lane.getLastStepVehicleNumber(departure_lane)
            outgoing_lanes_state.append(depart_number)
        to_return = np.array(phase_id + list(incoming_lanes_state) + list(outgoing_lanes_state)).reshape(self.state_dim)
        return to_return

    def observation_space(self):
        """Subclasses must override this method."""
        return spaces.Box(
            low=0,
            high=500,
            shape=self.state_dim
        )

class CustomObservation2(ObservationFunction):

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
        self.state_dim = (10, )
        
            

    def __call__(self):
        """Subclasses must override this method."""
        queue = [traci.lane.getLastStepHaltingNumber(lane) for lane in self.ts.lanes]
        return queue

    def observation_space(self):
        """Subclasses must override this method."""
        return spaces.Box(
            low=0,
            high=500,
            shape=self.state_dim
        )

class QueueObservation(ObservationFunction):

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
        self.state_dim = (len(self.incoming_lanes), )
        self.intersection_pos = traci.junction.getPosition(self.ts.id)
        
    def __call__(self):
        """Subclasses must override this method."""
        new_state = []
        for index, arrival_lane in enumerate(self.incoming_lanes):
            arrival_lane_vehicles = self.get_lane_vehicles(lane_id=arrival_lane)
            depart_lane_vehicles = 0
            for departure_lane in self.outgoing_lanes[index]:
                depart_lane_vehicles += self.get_lane_vehicles(lane_id=departure_lane)
            new_state.append(arrival_lane_vehicles-depart_lane_vehicles)
        return new_state
    
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


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
    
random.seed(10)

class CustomSUMORLEnv(SumoEnvironment):

    def __init__(self,
        net_file: str,
        route_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        virtual_display: Tuple[int, int] = (3200, 1800),
        begin_time: int = 0,
        num_seconds: int = 20000,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        single_agent: bool = False,
        reward_fn: Union[str, Callable, dict] = "diff-waiting-time",
        observation_class: ObservationFunction = DefaultObservationFunction,
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = "random",
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        render_mode: Optional[str] = None,
        new_traffic_lights: list = None,
        encode_function = None,
        traffic_flow_choices=[300,200],
        random_flow = False,
        real_data_type = False,
        percentage_added=None) -> None:
        super().__init__(net_file, route_file, out_csv_name, use_gui, virtual_display, begin_time, num_seconds, max_depart_delay, waiting_time_memory, time_to_teleport, delta_time, yellow_time, min_green, max_green, single_agent, reward_fn, observation_class, add_system_info, add_per_agent_info, sumo_seed, fixed_ts, sumo_warnings, additional_sumo_cmd, render_mode)
        if not encode_function == None:
            self.encode = encode_function
        self.vehicle_distribution = {}
        self.df = pd.DataFrame({})
        self.turn_vehicles = defaultdict(list)
        self.flow_data = defaultdict(list)
        self.dataframes = defaultdict(lambda: pd.DataFrame({}, columns=["time", "veh_id"]))
        self.vehicles_arrived_area_detector = defaultdict(list)
        self.vehicles_arrived_loop_detector = defaultdict(list)
        self.loop_detectors_edges = {}
        self.routes_start_with_edge = {}
        self.vehicles_flow = traffic_flow_choices
        self.random_flow = random_flow
        self.real_data_type = real_data_type
        if real_data_type:
            self.percentage_added = percentage_added
            self.load_real_data_detectors()
        self.vehicles_waiting_time = defaultdict(lambda: 0)

             
    def get_vehicles_distribution(self, flow_rate):
        np.random.seed(10)
        time_step = 1/3600  # Time step in hours (1 second)
        expected_arrivals_per_step = flow_rate * time_step
        num_vehicles_arrived_area_detector = np.array(list(map(round, np.random.exponential(expected_arrivals_per_step, self.sim_max_time))))
        return num_vehicles_arrived_area_detector

    def generate_distributation(self, step = 0, times_to_generate=[3600]):
        if len(self.vehicle_distribution) == 0:
            for route in traci.route.getIDList():
                rand = random.choice(self.vehicles_flow)
                self.vehicle_distribution[route] = self.get_vehicles_distribution(flow_rate=rand)

        for time in times_to_generate:
            if step == time:
                for route in self.vehicle_distribution.keys():
                    rand = random.choice(self.vehicles_flow)
                    self.vehicle_distribution[route] = self.get_vehicles_distribution(flow_rate=rand)
    
    def add_exponential_vehicle(self, step):
        step = int(step)
        self.generate_distributation(0, times_to_generate=[3600])
        entered = 0
        for route in traci.route.getIDList():
            if route not in self.vehicle_distribution.keys():
                self.vehicle_distribution[route] = self.get_vehicles_distribution(flow_rate=random.choice(self.vehicles_flow))
            for _ in range(self.vehicle_distribution[route][step]):
                veh_id = f'veh{self.random_word_generator()}'
                traci.vehicle.add(veh_id, routeID=route)
                self.flow_data[route].append({"time": traci.simulation.getTime(), "veh_id": veh_id})
                entered += self.vehicle_distribution[route][step]
    
    def _get_system_info(self):
        vehicles = traci.vehicle.getIDList()
        speeds = [traci.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = []
        for vehicle in vehicles:
            veh_waiting_time = traci.vehicle.getWaitingTime(vehicle)
            waiting_times.append(veh_waiting_time)
            self.vehicles_waiting_time[vehicle] = veh_waiting_time

        ts_dict = {}
        for ts in traci.trafficlight.getIDList():
            ts_dict["phase_" + ts] = self.traffic_signals[ts].get_total_queued()
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_time": traci.simulation.getTime(),
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
        } | ts_dict



    
    
    def random_word_generator(self, length = 10):
        return "".join(random.sample(string.ascii_letters, length))
    
    def _sumo_step(self):
        if len(self.loop_detectors_edges) == 0:
            self.get_loop_detecores_edges()
        # self.get_veh_arrived_to_ts()
        if self.random_flow:
            self.add_exponential_vehicle(traci.simulation.getTime())
        if self.real_data_type:
            self.real_data_add()
            pass
        self.df = pd.concat([self.df, pd.DataFrame([self._get_system_info()])], ignore_index=True)
        traci.simulationStep()
        
        # self.detect_area()
        # self.loop_detector()
    
    def custom_save_data(self, path, file_name):
        if not os.path.isdir(path):
            Path(path).mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path + "/" + file_name)
        with open(os.path.join(path, Path(file_name).stem + "_veh_waiting_time.json"), "w+") as f:
            json.dump(self.vehicles_waiting_time, f)

        self.df = pd.DataFrame({})

    def save_metadata(self, metadata:dict,path, file_name):
        if not os.path.isdir(path):
            Path(path).mkdir(parents=True, exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(metadata, f)
        
    def flow_data_save(self, path, file_name):
        if not os.path.isdir(path):
            os.mkdir(path)
        
        with open(path + "/" + file_name, "w+") as f:
            json.dump(self.flow_data, f, indent=2)
        self.flow_data = defaultdict(list)
    
    def save_turn_vehicles(self, path, file_name):
        if not os.path.isdir(path):
            os.mkdir(path)
        
        with open(path + "/" + file_name, "w+") as f:
            json.dump(self.turn_vehicles, f, indent=2)
        self.turn_vehicles = defaultdict(list)


    def check_vehicle_turn(self, vehicle_route, turn_routes=list(range(8,24))):
        if vehicle_route in turn_routes:
            return True
        else:
            return False
    
    def add_turn_vehicle(self, turn_vehicle_id, route_id):
        self.turn_vehicles[route_id].append({"time": traci.simulation.getTime(), "vehicle_id": turn_vehicle_id})
    
    def get_veh_arrived_to_ts(self, distance_threshold=10):
        loaded_vehicles_id = traci.vehicle.getLoadedIDList()
        traffic_id_lists = traci.trafficlight.getIDList()
        for vehicle_id in loaded_vehicles_id:
            for traffic_id in traffic_id_lists:
                if vehicle_id not in self.dataframes[traffic_id].veh_id:
                    traffic_pos = traci.junction.getPosition(traffic_id)
                    vehicle_position = traci.vehicle.getPosition(vehicle_id)
                    distance = traci.simulation.getDistance2D(vehicle_position[0], vehicle_position[1], traffic_pos[0], traffic_pos[1])
                    if distance < distance_threshold:
                        self.dataframes[traffic_id] = pd.concat([self.dataframes[traffic_id], pd.DataFrame({'time': [traci.simulation.getTime()], 'veh_id': [vehicle_id]})])
                        break
    
    def detect_area(self):
        for e2_id in traci.lanearea.getIDList():
            self.vehicles_arrived_area_detector[e2_id].append(traci.lanearea.getLastStepVehicleNumber(e2_id))

    def loop_detector(self):
        for loop_id in traci.inductionloop.getIDList():
            self.vehicles_arrived_loop_detector[loop_id].append(traci.inductionloop.getLastStepVehicleNumber(loop_id))
    
    def get_loop_detecores_edges(self):
        for loop_id in traci.inductionloop.getIDList():
            self.loop_detectors_edges[loop_id] = traci.lane.getEdgeID(traci.inductionloop.getLaneID(loop_id))
        for loop_id, edge_id in self.loop_detectors_edges.items():
            self.routes_start_with_edge[edge_id] = [route for route in traci.route.getIDList() if traci.route.getEdges(route)[0] == edge_id]
        
        with open("loop_detectors_edges.json", "w") as f:
            json.dump(self.loop_detectors_edges, f)
        
        with open("routes_start_with_edge.json", "w") as f:
            json.dump(self.routes_start_with_edge, f)
        
    def load_real_data_detectors(self, window=10):
        self.real_data_detector = pd.read_csv("/Users/kiani014/Documents/Github/FRITS2/learning_controller/output/i4-real_data_results/_loopdetector_rl_test.csv")
        self.real_data_detector = self.real_data_detector.iloc[:, 1:]

        for column in self.real_data_detector.columns:
            summation_list = self.get_summation(self.real_data_detector, column, window)
            summation_list = [item for item in summation_list for _ in range(window)]
            self.real_data_detector[column] = summation_list

    def real_data_add(self):
        partition_data = self.real_data_detector.iloc[[traci.simulation.getTime()],:]
        partition_data = partition_data.loc[:, (partition_data!=0).any(axis=0)]
        if len(self.loop_detectors_edges) == 0:
            return
        for column in partition_data.columns:
            no_vehicles_to_be_added = (1+self.percentage_added) * partition_data[column].values[0]
            edge_id = self.loop_detectors_edges[column]
            route_id = random.choice(self.routes_start_with_edge[edge_id])
            for _ in range(int(round(no_vehicles_to_be_added))):
                veh_id = f'veh{self.random_word_generator()}'
                traci.vehicle.add(veh_id, routeID=route_id)
                self.flow_data[route_id].append({"time": traci.simulation.getTime(), "veh_id": veh_id})

    def get_traffic_signal_green_phase(self, traffic_signal_id):
        return self.traffic_signals[traffic_signal_id].green_phase
    
    def get_summation(self, data, column_name, window):
        to_return = []
        for i in range(data.shape[0]//window):
            group = data[column_name].iloc[i:i+window]
            group_average = group.sum()
            to_return.append(group_average)
        return to_return

    
    def save_veh_arrived_to_ts(self, path, file_name, epsiode):
        if not os.path.isdir(path):
            os.mkdir(path)
        for key, dataframe in self.dataframes.items():
            dataframe.to_csv(path + "/" + key + "_" +str(epsiode) + "_" + file_name)
        
        pd.DataFrame(self.vehicles_arrived_area_detector).to_csv(path + "/" + "_detectors_" + str(epsiode) + "_" + file_name)
        
        self.vehicles_arrived_area_detector = defaultdict(list)
        
        self.dataframe = defaultdict(list)
    
    def save_loopdetector_data(self, path, file_name):
        if not os.path.isdir(path):
            os.mkdir(path)
        pd.DataFrame(self.vehicles_arrived_loop_detector).to_csv(path + "/" + "_loopdetector_" + "_".join(list(map(str, self.vehicles_flow))) + "_" + file_name)
        self.vehicles_arrived_loop_detector = defaultdict(list)
    
    def delete_cache(self):
        self.vehicle_distribution = {}
        self.df = pd.DataFrame({})
        self.turn_vehicles = defaultdict(list)
        self.flow_data = defaultdict(list)
        self.dataframes = defaultdict(lambda: pd.DataFrame({}, columns=["time", "veh_id"]))
        self.vehicles_arrived_area_detector = defaultdict(list)
        self.vehicles_arrived_loop_detector = defaultdict(list)