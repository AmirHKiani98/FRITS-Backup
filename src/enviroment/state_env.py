import random
import numpy as np
from sumo_rl.environment.observations import ObservationFunction
from src.enviroment.traffic_signal import TrafficSignalCustom
from gym import spaces
import traci

def create_arrival_departure_state(alpha=0.0, noise_added=False):
    """Factory function that returns ArrivalDepartureState class with preset parameters"""
    class ConfiguredArrivalDepartureState(ArrivalDepartureState):
        def __init__(self, ts: TrafficSignalCustom):
            super().__init__(ts, alpha=alpha, noise_added=noise_added)
    
    return ConfiguredArrivalDepartureState

class ArrivalDepartureState(ObservationFunction):
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
        
        state = arrival + departure
        
        # Apply noise if enabled
        if self.noise_added and self.alpha > 0:
            state = self.add_noise_to_state(state)
            
        print(f"Arrival: {arrival}, Departure: {departure}, Alpha: {self.alpha}, Noise: {self.noise_added}, Ts Id: {self.ts.id}, State: {state}")
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
        """Subclasses must override this method."""
        return spaces.Box(
            low=0,
            high=500,
            shape=self.state_dim
        )
    