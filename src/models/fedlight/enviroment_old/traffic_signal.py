from typing import Callable, Union
from sumo_rl.environment.traffic_signal import TrafficSignal


class TrafficSignalCustom(TrafficSignal):
    def __init__(self, env,
                 ts_id: str,
                 delta_time: int,
                 yellow_time: int,
                 min_green: int,
                 max_green: int,
                 begin_time: int,
                 reward_fn: Union[str, Callable],
                 sumo):
        """Initializes a TrafficSignalCustom object."""
        super().__init__(env, ts_id, delta_time, yellow_time, min_green, max_green, begin_time, reward_fn, sumo)
        self.previous_green_phase = None

    def set_next_phase(self, new_phase: int):
        """Set the next phase of the traffic signal."""
        self.previous_green_phase = self.green_phase
        super().set_next_phase(new_phase)
    
    def get_previous_green_phase(self):
        return self.previous_green_phase