import os
import sys
import sumolib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci


@dataclass
class SimulationConfig:
    """Configuration class for SUMO simulation parameters."""
    simulation_time: int
    net_file: str
    route_file: str
    output_file: str
    use_gui: bool
    step_length: float

    @staticmethod
    def from_args(args) -> "SimulationConfig":
        """Create SimulationConfig from argparse arguments."""
        return SimulationConfig(
            simulation_time=args.simulation_time,
            net_file=args.net_file,
            route_file=args.route_file,
            output_file=args.output_file,
            use_gui=args.use_gui,
            step_length=args.step_length
        )


class SUMOController:
    """Handles SUMO simulation initialization and control."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.script_path = Path(__file__).parent.absolute()
        self.sumo_binary = sumolib.checkBinary("sumo-gui" if config.use_gui else "sumo")
        self._simulation_started = False
    
    def _build_sumo_command(self) -> List[str]:
        """Build the SUMO command line arguments."""
        net_path = self.script_path / self.config.net_file
        route_path = self.script_path / self.config.route_file
        
        cmd = [
            self.sumo_binary,
            "-n", str(net_path),
            "-r", str(route_path),
            "--step-length", str(self.config.step_length)
        ]
        
        if not self.config.use_gui:
            cmd.extend(["--no-warnings", "--no-step-log"])
        
        return cmd
    
    def start_simulation(self) -> None:
        """Start the SUMO simulation."""
        if self._simulation_started:
            raise RuntimeError("Simulation already started")
        
        sumo_cmd = self._build_sumo_command()
        traci.start(sumo_cmd)
        self._simulation_started = True
    
    def step_simulation(self) -> None:
        """Advance simulation by one step."""
        if not self._simulation_started:
            raise RuntimeError("Simulation not started")
        traci.simulationStep()
    
    def close_simulation(self) -> None:
        """Close the SUMO simulation."""
        if self._simulation_started:
            traci.close()
            self._simulation_started = False
    
    def __enter__(self):
        self.start_simulation()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_simulation()


class TrafficDataCollector:
    """Collects and processes traffic data from SUMO simulation."""
    
    @staticmethod
    def get_vehicle_data() -> Dict:
        """Get current vehicle data from simulation."""
        vehicles = traci.vehicle.getIDList()
        speeds = [traci.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [traci.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        
        return {
            "vehicles": vehicles,
            "speeds": speeds,
            "waiting_times": waiting_times
        }
    
    @staticmethod
    def get_traffic_light_data() -> Dict[str, Dict]:
        """Get traffic light data from simulation."""
        traffic_lights = traci.trafficlight.getIDList()
        ts_data = {}
        
        for ts_id in traffic_lights:
            ts_data[ts_id] = {
                "current_phase": traci.trafficlight.getPhase(ts_id),
                "next_switch": traci.trafficlight.getNextSwitch(ts_id),
                "controlled_lanes": traci.trafficlight.getControlledLanes(ts_id),
            }
        
        return ts_data
    
    def get_system_metrics(self) -> Dict:
        """Calculate system-wide traffic metrics."""
        vehicle_data = self.get_vehicle_data()
        vehicles = vehicle_data["vehicles"]
        speeds = vehicle_data["speeds"]
        waiting_times = vehicle_data["waiting_times"]
        
        # Calculate metrics
        total_vehicles = len(vehicles)
        stopped_vehicles = sum(1 for speed in speeds if speed < 0.1)
        total_waiting_time = sum(waiting_times)
        mean_waiting_time = np.mean(waiting_times) if waiting_times else 0.0
        mean_speed = np.mean(speeds) if speeds else 0.0
        
        return {
            "system_time": traci.simulation.getTime(),
            "system_total_stopped": stopped_vehicles,
            "system_total_waiting_time": total_waiting_time,
            "system_mean_waiting_time": mean_waiting_time,
            "system_mean_speed": mean_speed,
            "system_total_vehicles": total_vehicles,
        }


class DataLogger:
    """Handles data logging and file output."""
    
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.data_records: List[Dict] = []
    
    def add_record(self, record: Dict) -> None:
        """Add a data record to the logger."""
        self.data_records.append(record.copy())
    
    def save_to_csv(self) -> None:
        """Save collected data to CSV file."""
        if not self.data_records:
            print("No data to save")
            return
        
        df = pd.DataFrame(self.data_records)
        df.to_csv(self.output_file, index=False)
        print(f"Data saved to {self.output_file}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get data as pandas DataFrame."""
        return pd.DataFrame(self.data_records)


class FixedTimingSimulation:
    """Main simulation class that orchestrates the fixed timing simulation."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.controller = SUMOController(config)
        self.data_collector = TrafficDataCollector()
        self.logger = DataLogger(config.output_file)
    
    def run_simulation(self) -> pd.DataFrame:
        """Run the complete simulation and return results."""
        print(f"Starting fixed timing simulation for {self.config.simulation_time} steps...")
        
        with self.controller:
            # Collect initial data
            initial_metrics = self.data_collector.get_system_metrics()
            self.logger.add_record(initial_metrics)
            
            # Run simulation steps
            for step in range(self.config.simulation_time):
                if step % 100 == 0:  # Print progress every 100 steps
                    print(f"Step {step}/{self.config.simulation_time}")
                
                self.controller.step_simulation()
                
                # Collect metrics
                metrics = self.data_collector.get_system_metrics()
                self.logger.add_record(metrics)
        
        # Save results
        self.logger.save_to_csv()
        print("Simulation completed successfully!")
        
        return self.logger.get_dataframe()


def create_default_config(
    simulation_time: int = 2700,
    net_file: str = "net/4x4.net.xml",
    route_file: str = "rou/4x4c2c1.rou.xml",
    output_file: str = "4x4_fixed.csv",
    use_gui: bool = False,
    step_length: float = 1.0
) -> SimulationConfig:
    """Create a default simulation configuration."""
    return SimulationConfig(
        simulation_time=simulation_time,
        net_file=net_file,
        route_file=route_file,
        output_file=output_file,
        use_gui=use_gui,
        step_length=step_length
    )


def main():
    """Main function to run the fixed timing simulation."""
    # Create configuration
    config = create_default_config()
    
    # Run simulation
    simulation = FixedTimingSimulation(config)
    results_df = simulation.run_simulation()
    
    # Optional: Print summary statistics
    print("\n=== Simulation Summary ===")
    print(f"Total simulation time: {config.simulation_time} seconds")
    print(f"Average vehicles: {results_df['system_total_vehicles'].mean():.2f}")
    print(f"Average stopped vehicles: {results_df['system_total_stopped'].mean():.2f}")
    print(f"Average waiting time: {results_df['system_mean_waiting_time'].mean():.2f} seconds")
    print(f"Average speed: {results_df['system_mean_speed'].mean():.2f} m/s")


if __name__ == "__main__":
    main()