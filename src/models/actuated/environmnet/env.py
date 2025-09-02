
import sumolib
import traci
import os
import random
import traci
from src.models.actuated.environmnet.traffic_signal import ActuatedTLS 
import numpy as np
from collections import defaultdict
import pandas as pd
class ActuateEnv:
    def __init__(self, route_path, net_path, min_green=6.0, max_green=45.0, t_crit=2.5, store_results_path="src/models/actuated/output", alpha = 0, attacked_intersections=""):
        self.route_path = route_path
        self.net_path = net_path
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.t_crit = float(t_crit)
        self.store_results_path = store_results_path
        self.vehicles_waiting_time = defaultdict(lambda: 0)
        self.loop_detectors_path = self.generate_loop_detectors()
        self.alpha = alpha
        self.df = pd.DataFrame()
        self.attacked_intersections = attacked_intersections.split(",") if attacked_intersections else []

    def generate_loop_detectors(self):
        # GGet SUMO_HOME Path
        SUMO_HOME = os.environ.get('SUMO_HOME')
        if SUMO_HOME is None:
            raise EnvironmentError("Please declare the 'SUMO_HOME' environment variable.")
        # tools\output\generateTLSE1Detectors.py -n network --frequency 10 -o .src/additionals/random_name.add.xml
        random_name = f"random_{random.randint(1000, 9999)}.add.xml"
        output_path = os.path.join('src', 'additionals', random_name)
        os.system(
            f"python '{os.path.join(SUMO_HOME, 'tools', 'output', 'generateTLSE1Detectors.py')}' -n '{self.net_path}' --frequency 10 -d 10 -o '{output_path}'"
        )
        return output_path

    def start(self, gui=False):
        sumo_bin = sumolib.checkBinary("sumo-gui" if gui else "sumo")
        traci.start([sumo_bin, "-n", self.net_path, "-r", self.route_path, "-a", self.loop_detectors_path, "--quit-on-end", "--start"])
        # build controllers for every TLS
        self.tls_list = [
            ActuatedTLS(tls_id, t_min=self.min_green, t_max=self.max_green, t_crit=self.t_crit,
                        lane_to_loop=lambda ln: f"e1det_{ln}", alpha=self.alpha, is_attacked=(tls_id in self.attacked_intersections))
            for tls_id in traci.trafficlight.getIDList()
        ]
    
    def _get_system_info(self):
        vehicles = traci.vehicle.getIDList()
        speeds = [traci.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = []
        for vehicle in vehicles:
            veh_waiting_time = traci.vehicle.getWaitingTime(vehicle)
            waiting_times.append(veh_waiting_time)
            self.vehicles_waiting_time[vehicle] = int(veh_waiting_time) if isinstance(veh_waiting_time, (int, float, str)) else 0

        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_time": traci.simulation.getTime(),
            "system_total_stopped": sum(int(float(speed) < 0.1) for speed in speeds if isinstance(speed, (int, float, str))),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
        }
    def step(self):
        traci.simulationStep()
        for tls in self.tls_list:
            tls.step()
        self.df = pd.concat([self.df, pd.DataFrame([self._get_system_info()])], ignore_index=True)

    def store_results(self):
        if os.path.dirname(self.store_results_path) and not os.path.exists(os.path.dirname(self.store_results_path)):
            os.makedirs(os.path.dirname(self.store_results_path), exist_ok=True)

        self.df.to_csv(self.store_results_path, index=False)

    def close(self):
        traci.close()
