import traci
import random
class ActuatedTLS:
    def __init__(self, tls_id, t_min=6.0, t_max=45.0, t_crit=2.5, lane_to_loop=None, warn_missing=True, alpha=0, is_attacked=False):
        self.id = tls_id
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.t_crit = float(t_crit)
        self.warn_missing = warn_missing
        self.alpha = alpha
        self.last_detection = float("-inf")
        
        # TLS link index -> in-lanes (from SUMO)
        cl = traci.trafficlight.getControlledLinks(self.id)
        if not cl:
            raise ValueError(f"{tls_id}: no controlled links")
        self.linkidx_to_inlanes = {i: list({inLn for (inLn, _, _) in conns}) for i, conns in enumerate(cl)}

        self.lane_to_loop = lane_to_loop or (lambda lane: f"e1det_{lane}")

        self.phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0].getPhases()
        self.last_phase = traci.trafficlight.getPhase(self.id)
        self.green_start_time = traci.simulation.getTime()
        self._missing_warned = set()
        self.is_attacked = is_attacked
        

    def _state(self):
        return traci.trafficlight.getRedYellowGreenState(self.id)

    def _lanes_with_G(self):
        state = self._state()
        lanes = set()
        for i, ch in enumerate(state):
            if ch == 'G':
                lanes.update(self.linkidx_to_inlanes.get(i, ()))
        return lanes

    def _loops_active(self, lanes):
        any_active = False
        for lane in lanes:
            det = self.lane_to_loop(lane)
            try:
                indicated_veh = traci.inductionloop.getLastStepVehicleNumber(det)
                if not isinstance(indicated_veh, (int, float)):
                    raise ValueError(f"{self.id}: invalid gap {indicated_veh} for loop {det}")
                if indicated_veh > 0:
                    gap = 0
                    self.last_detection = traci.simulation.getTime()
                else:
                    gap = traci.simulation.getTime() - self.last_detection
                if self.is_attacked:
                    uniform_value = random.uniform(0, 1)
                    alpha_posibility = self.alpha/10
                    if uniform_value < alpha_posibility:
                        gap = 0
            except traci.TraCIException:
                if self.warn_missing and det not in self._missing_warned:
                    self._missing_warned.add(det)
                    print(f"[WARN] {self.id}: missing loop {det} for lane {lane}")
                continue
            if not isinstance(gap, (int, float)):
                raise ValueError(f"{self.id}: invalid gap {gap} for loop {det}")
            if gap <= self.t_crit:
                any_active = True
                break
        return any_active

    def _phase_elapsed(self, now):
        ph = traci.trafficlight.getPhase(self.id)
        if ph != self.last_phase:
            self.last_phase = ph
            self.green_start_time = now
        return now - self.green_start_time

    def _next_yellow_after(self, green_idx):
        n = len(self.phases)
        for step in range(1, n+1):
            i = (green_idx + step) % n
            if 'y' in self.phases[i].state:
                return i
        return (green_idx + 1) % n  # fallback

    def step(self):
        now = traci.simulation.getTime()
        ph = traci.trafficlight.getPhase(self.id)
        state = self._state()

        # skip during change or all-red
        if ('y' in state) or all(ch == 'r' for ch in state):
            self._phase_elapsed(now)
            return

        G = self._phase_elapsed(now)
        if G < self.t_min:
            return
        if G >= self.t_max:
            traci.trafficlight.setPhase(self.id, self._next_yellow_after(ph))
            return

        lanes_G = self._lanes_with_G()
        if not lanes_G:
            # no controlling lanes flagged G in this step; do nothing
            return

        if self._loops_active(lanes_G):
            return  # keep green
        else:
            traci.trafficlight.setPhase(self.id, self._next_yellow_after(ph))