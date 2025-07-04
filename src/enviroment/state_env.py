import random
import numpy as np
from sumo_rl.environment.observations import ObservationFunction
from src.enviroment.traffic_signal import TrafficSignalCustom
from gym import spaces
import traci

class ConfigurableArrivalDepartureState(ObservationFunction):
    """
    A configurable arrival-departure state observation class that supports noise injection.
    This is a top-level class that can be pickled for multiprocessing.
    """
    def __init__(self, ts: TrafficSignalCustom, alpha=0.0, noise_added=False):
        super().__init__(ts)
        
        # Store the noise parameters
        self.alpha = alpha
        self.noise_added = noise_added
 
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
        """Calculate the arrival-departure state with optional noise."""
        arrival = []
        departure = []
        for index, arrival_lane in enumerate(self.incoming_lanes):
            arrival_lane_vehicles = self.get_lane_vehicles(lane_id=arrival_lane)
            depart_lane_vehicles = 0
            for departure_lane in self.outgoing_lanes[index]:
                depart_lane_vehicles += self.get_lane_vehicles(lane_id=departure_lane)
            arrival.append(arrival_lane_vehicles)
            departure.append(depart_lane_vehicles)
        
        state = arrival + departure
        
        # Apply noise if enabled
        if self.noise_added and self.alpha > 0:
            state = self.add_noise_to_state(state)
            
        return state
    
    def add_noise_to_state(self, state):
        """Add noise to the state based on alpha value"""
        noisy_state = []
        for value in state:
            # Generate noise based on alpha
            noise = random.randint(0, int(self.alpha))
            noisy_value = max(0, value + noise)  # Ensure non-negative values
            noisy_state.append(int(noisy_value))
        return noisy_state
    
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
            if isinstance(distance, (int, float)) and distance < distance_threshold:
                summation += 1
        
        return summation

    def observation_space(self):
        """Return the observation space for this state representation."""
        return spaces.Box(
            low=0,
            high=500,
            shape=self.state_dim
        )

class ObservationClassFactory:
    """
    A pickleable factory class for creating observation classes with specific parameters.
    This can be passed to multiprocessing workers.
    """
    def __init__(self, alpha=0.0, noise_added=False):
        self.alpha = alpha
        self.noise_added = noise_added
    
    def create_observation_class(self, ts):
        """Create an observation class instance with the stored parameters."""
        return ConfigurableArrivalDepartureState(ts, alpha=self.alpha, noise_added=self.noise_added)
    
    def __call__(self, ts):
        """Make the factory callable."""
        return self.create_observation_class(ts)

def create_arrival_departure_state(alpha=0.0, noise_added=False):
    """
    Factory function that returns an ObservationClassFactory.
    This approach is pickleable and works with multiprocessing.
    """
    return ObservationClassFactory(alpha=alpha, noise_added=noise_added)

