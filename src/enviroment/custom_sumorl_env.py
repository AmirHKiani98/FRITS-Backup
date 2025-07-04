"""
Custom SUMO Reinforcement Learning Environment
"""
import random
import string
import os
from pathlib import Path
import json
from collections import defaultdict
from typing import Callable, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sumo_rl.environment.traffic_signal import TrafficSignal
from sumo_rl.environment.observations import DefaultObservationFunction, ObservationFunction
from sumo_rl import SumoEnvironment
import traci

from src.models.fedlight.enviroment.traffic_signal import TrafficSignalCustom
def empty_vehicle_df():
    return pd.DataFrame({}, columns=["time", "veh_id"])
LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

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
        observation_class: ObservationFunction = DefaultObservationFunction, # type: ignore
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = "random",
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        render_mode: Optional[str] = None,
        new_traffic_lights: Optional[list] = None,
        encode_function = None,
        traffic_flow_choices=[300,200],
        random_flow = False,
        real_data_type = False,
        percentage_added=None,
        step_length=0.1) -> None:
        super().__init__(net_file, route_file, out_csv_name, use_gui, virtual_display, begin_time, num_seconds, max_depart_delay, waiting_time_memory, time_to_teleport, delta_time, yellow_time, min_green, max_green, single_agent, reward_fn, observation_class, add_system_info, add_per_agent_info, sumo_seed, fixed_ts, sumo_warnings, additional_sumo_cmd, render_mode)
        if not encode_function == None:
            self.encode = encode_function
        self.vehicle_distribution = {}
        self.df = pd.DataFrame({})
        self.turn_vehicles = defaultdict(list)
        self.flow_data = defaultdict(list)
        self.dataframes = defaultdict(empty_vehicle_df)
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
        self.step_length = step_length

             
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
            self.vehicles_waiting_time[vehicle] = int(veh_waiting_time) if isinstance(veh_waiting_time, (int, float, str)) else 0

        ts_dict = {}
        for ts in traci.trafficlight.getIDList():
            ts_dict["phase_" + ts] = self.traffic_signals[ts].get_total_queued()
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_time": traci.simulation.getTime(),
            "system_total_stopped": sum(int(float(speed) < 0.1) for speed in speeds if isinstance(speed, (int, float, str))),
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
                    if isinstance(distance, (int, float)) and isinstance(distance, (int, float)) and distance < distance_threshold:
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
        partition_data = self.real_data_detector.iloc[[traci.simulation.getTime()],:] # type: ignore
        partition_data = partition_data.loc[:, (partition_data!=0).any(axis=0)]
        if len(self.loop_detectors_edges) == 0:
            return
        for column in partition_data.columns:
            no_vehicles_to_be_added = (1+self.percentage_added) * partition_data[column].values[0] # type: ignore
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
    

    def  reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        
        # Override traffic signal class
        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: TrafficSignalCustom(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn[ts],
                    self.sumo,
                )
                for ts in self.reward_fn.keys()
            }
        else:
            self.traffic_signals = {
                ts: TrafficSignalCustom(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn,
                    self.sumo,
                )
                for ts in self.ts_ids
            }

        self.vehicles = dict()
        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()
        else:
            return self._compute_observations()
    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-n",
            self._net,
            "-r",
            self._route,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
            "--no-step-log",
            "--step-length", str(self.step_length),
            "--no-warnings",
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            ## Following part is erronous
            # if self.render_mode == "rgb_array":
            #     sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
            #     from pyvirtualdisplay.smartdisplay import SmartDisplay

            #     print("Creating a virtual display.")
            #     self.disp = SmartDisplay(size=self.virtual_display)
            #     self.disp.start()
            #     print("Virtual display started.")


        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui or self.render_mode is not None:
            if "DEFAULT_VIEW" not in dir(traci.gui):  # traci.gui.DEFAULT_VIEW is not defined in libsumo
                traci.gui.DEFAULT_VIEW = "View #0"
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")