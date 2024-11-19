import os
import sys
import sumolib
import pandas as pd
import numpy as np

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

simulation_time = 2700

sumo_binary = sumolib.checkBinary("sumo")


script_path = os.path.dirname(os.path.abspath(__file__))

rou = script_path+"/rou/d_1_turnCount_am_offpeak_1.rou.xml"
net = script_path+"/net/i4_new.net.xml"

sumo_cmd = [sumo_binary, "-n", net, "-r", rou]

traci.start(sumo_cmd)

def _get_system_info():
        vehicles = traci.vehicle.getIDList()
        speeds = [traci.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = []
        for vehicle in vehicles:
            veh_waiting_time = traci.vehicle.getWaitingTime(vehicle)
            waiting_times.append(veh_waiting_time)

        ts_dict = {}
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_time": traci.simulation.getTime(),
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
            "system_total_vehicles": len(vehicles),
        } | ts_dict

df = pd.DataFrame({})

pd.concat([df, pd.DataFrame([_get_system_info()])], ignore_index=True)
for _ in range(simulation_time):
    traci.simulationStep()
    df = pd.concat([df, pd.DataFrame([_get_system_info()])], ignore_index=True)

df.to_csv("d_1_fixed.csv")