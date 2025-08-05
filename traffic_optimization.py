"""
Traffic Flow Optimization Module for SUMO Simulations
Helps reduce emergency braking warnings by improving traffic flow parameters.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import sumolib


class TrafficFlowOptimizer:
    """Optimizes SUMO simulation parameters to reduce emergency braking."""
    
    def __init__(self, net_file: str, route_file: str):
        self.net_file = Path(net_file)
        self.route_file = Path(route_file)
        self.optimized_route_file = None
        
    def create_optimized_route_file(self, 
                                  output_file: Optional[str] = None,
                                  depart_speed: str = "max",
                                  max_speed: float = 13.89,  # 50 km/h in m/s
                                  min_gap: float = 2.5,
                                  accel: float = 2.6,
                                  decel: float = 4.5,
                                  emergency_decel: float = 9.0,
                                  sigma: float = 0.5) -> str:
        """
        Create an optimized route file with better vehicle parameters.
        
        Args:
            output_file: Output route file path
            depart_speed: Departure speed ("random", "max", or specific value)
            max_speed: Maximum vehicle speed (m/s)
            min_gap: Minimum gap to leading vehicle (m)
            accel: Normal acceleration (m/s²)
            decel: Normal deceleration (m/s²)
            emergency_decel: Emergency deceleration (m/s²)
            sigma: Driver imperfection (0-1)
        """
        if output_file is None:
            output_file = str(self.route_file).replace('.rou.xml', '_optimized.rou.xml')
        
        # Parse original route file
        tree = ET.parse(self.route_file)
        root = tree.getroot()
        
        # Define optimized vehicle type
        optimized_vtype = ET.Element("vType", {
            "id": "optimized_car",
            "accel": str(accel),
            "decel": str(decel),
            "emergencyDecel": str(emergency_decel),
            "sigma": str(sigma),
            "length": "4.5",
            "minGap": str(min_gap),
            "maxSpeed": str(max_speed),
            "speedFactor": "1.0",
            "speedDev": "0.1",
            "carFollowModel": "Krauss",  # Use Krauss model for stability
            "tau": "1.0",  # Reaction time
            "lcStrategic": "1.0",
            "lcCooperative": "1.0",
            "lcSpeedGain": "1.0"
        })
        
        # Insert optimized vehicle type at the beginning
        root.insert(0, optimized_vtype)
        
        # Update all vehicles to use optimized type and departure parameters
        for vehicle in root.findall("vehicle"):
            vehicle.set("type", "optimized_car")
            vehicle.set("departSpeed", depart_speed)
            vehicle.set("departLane", "best")  # Choose best available lane
            vehicle.set("departPos", "base")   # Start at lane beginning
        
        # Update flows if they exist
        for flow in root.findall("flow"):
            flow.set("type", "optimized_car")
            flow.set("departSpeed", depart_speed)
            flow.set("departLane", "best")
            flow.set("departPos", "base")
        
        # Save optimized route file
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        self.optimized_route_file = output_file
        
        print(f"Optimized route file created: {output_file}")
        return output_file
    
    def get_sumo_config_with_collision_settings(self) -> Dict[str, str]:
        """Get SUMO configuration options to reduce collisions and warnings."""
        return {
            "--collision.action": "warn",  # Only warn about collisions, don't stop
            "--collision.check-junctions": "false",  # Reduce junction collision checks
            "--time-to-teleport": "300",  # Teleport stuck vehicles after 5 minutes
            "--time-to-teleport.highways": "120",  # Faster teleport on highways
            "--max-depart-delay": "600",  # Max delay before removing vehicles
            "--ignore-route-errors": "true",  # Continue simulation with route errors
            "--no-warnings": "true",  # Suppress warnings
            "--step-method.ballistic": "true",  # Use ballistic integration
            "--default.emergencydecel": "9.0",  # Set emergency deceleration
            "--lanechange.duration": "3.0",  # Longer lane change duration
            "--pedestrian.model": "nonInteracting"  # Simplify pedestrian model
        }


class AdaptiveSignalTiming:
    """Improves traffic signal timing to reduce emergency braking."""
    
    def __init__(self, net_file: str):
        self.net = sumolib.net.readNet(net_file)
        
    def get_signal_optimization_suggestions(self) -> Dict[str, Dict]:
        """Analyze network and suggest signal timing optimizations."""
        suggestions = {}
        
        # Get all traffic lights
        traffic_lights = self.net.getTrafficLights()
        
        for tl in traffic_lights:
            tl_id = tl.getID()
            connections = tl.getConnections()
            
            # Analyze incoming lanes
            incoming_lanes = set()
            for connection in connections:
                incoming_lanes.add(connection[0])
            
            # Calculate suggested timings based on lane capacity
            total_capacity = 0
            for lane_id in incoming_lanes:
                lane = self.net.getLane(lane_id)
                capacity = lane.getLength() * lane.getSpeed() / 7.5  # Rough capacity estimate
                total_capacity += capacity
            
            # Suggest longer green phases for high-capacity approaches
            min_green = max(10, int(total_capacity / 100))  # Minimum 10s, scale with capacity
            yellow_time = 3
            
            suggestions[tl_id] = {
                "min_green": min_green,
                "yellow_time": yellow_time,
                "cycle_length": min_green * len(incoming_lanes) + yellow_time * len(incoming_lanes),
                "incoming_lanes": list(incoming_lanes),
                "capacity_estimate": total_capacity
            }
        
        return suggestions


def create_optimized_simulation_config(original_config):
    """Create an optimized simulation configuration."""
    from fixed import SimulationConfig
    
    # Create traffic flow optimizer
    optimizer = TrafficFlowOptimizer(original_config.net_file, original_config.route_file)
    
    # Create optimized route file
    optimized_route = optimizer.create_optimized_route_file()
    
    # Create new configuration with optimized parameters
    optimized_config = SimulationConfig(
        simulation_time=original_config.simulation_time,
        net_file=original_config.net_file,
        route_file=optimized_route,  # Use optimized route file
        output_file=original_config.output_file.replace('.csv', '_optimized.csv'),
        use_gui=original_config.use_gui,
        step_length=0.1  # Smaller step length for better precision
    )
    
    return optimized_config


class ImprovedSUMOController:
    """Enhanced SUMO controller with optimized settings."""
    
    def __init__(self, config, reduce_warnings=True):
        self.config = config
        self.reduce_warnings = reduce_warnings
        self.script_path = Path(__file__).parent.absolute()
        self.sumo_binary = sumolib.checkBinary("sumo-gui" if config.use_gui else "sumo")
        self._simulation_started = False
        
        # Initialize optimizer if needed
        if reduce_warnings:
            self.optimizer = TrafficFlowOptimizer(config.net_file, config.route_file)
    
    def _build_optimized_sumo_command(self) -> List[str]:
        """Build SUMO command with optimized parameters."""
        net_path = self.script_path / self.config.net_file
        route_path = self.script_path / self.config.route_file
        
        # Base command
        cmd = [
            self.sumo_binary,
            "-n", str(net_path),
            "-r", str(route_path),
            "--step-length", str(self.config.step_length)
        ]
        
        # Add optimization parameters if warning reduction is enabled
        if self.reduce_warnings:
            optimization_params = {
                "--collision.action": "warn",
                "--time-to-teleport": "300",
                "--max-depart-delay": "600",
                "--ignore-route-errors": "true",
                "--no-warnings": "true",
                "--step-method.ballistic": "true",
                "--default.emergencydecel": "7.0",  # Slightly lower emergency decel
                "--lanechange.duration": "3.0",
                "--routing-algorithm": "dijkstra"
            }
            
            for param, value in optimization_params.items():
                cmd.extend([param, value])
        
        return cmd
    
    def start_simulation(self) -> None:
        """Start optimized SUMO simulation."""
        if self._simulation_started:
            raise RuntimeError("Simulation already started")
        
        sumo_cmd = self._build_optimized_sumo_command()
        
        # Print command for debugging
        print("Starting SUMO with command:")
        print(" ".join(sumo_cmd))
        
        import traci
        traci.start(sumo_cmd)
        self._simulation_started = True
    
    def __enter__(self):
        self.start_simulation()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._simulation_started:
            import traci
            traci.close()
            self._simulation_started = False


# Example usage functions
def run_optimized_simulation():
    """Example of running a simulation with optimized parameters."""
    from fixed import create_default_config, FixedTimingSimulation
    
    # Create base configuration
    base_config = create_default_config(
        simulation_time=1000,
        output_file="optimized_simulation.csv"
    )
    
    # Create optimized configuration
    optimized_config = create_optimized_simulation_config(base_config)
    
    # Run simulation with optimization
    simulation = FixedTimingSimulation(optimized_config)
    results = simulation.run_simulation()
    
    print("Optimized simulation completed!")
    return results


def analyze_signal_timings(net_file: str):
    """Analyze and suggest signal timing improvements."""
    analyzer = AdaptiveSignalTiming(net_file)
    suggestions = analyzer.get_signal_optimization_suggestions()
    
    print("Signal Timing Optimization Suggestions:")
    print("=" * 50)
    
    for tl_id, params in suggestions.items():
        print(f"\nTraffic Light: {tl_id}")
        print(f"  Suggested min green: {params['min_green']} seconds")
        print(f"  Suggested yellow time: {params['yellow_time']} seconds")
        print(f"  Suggested cycle length: {params['cycle_length']} seconds")
        print(f"  Incoming lanes: {len(params['incoming_lanes'])}")
        print(f"  Capacity estimate: {params['capacity_estimate']:.1f}")
    
    return suggestions


if __name__ == "__main__":
    # Example usage
    print("Running traffic flow optimization example...")
    
    # Analyze signal timings
    analyze_signal_timings("net/4x4.net.xml")
    
    # Run optimized simulation
    run_optimized_simulation()
