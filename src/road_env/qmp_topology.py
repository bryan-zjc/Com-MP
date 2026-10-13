"""Lane-based detector state and observed turns, without vehicle routes."""
from collections import defaultdict, deque
from .sumo_bootstrap import ensure_sumo_python_tools
ensure_sumo_python_tools()
import traci
from .env_CVMP import CVMPController, CVMPTracker, FixedTurningEstimator, VehicleSnapshot

class QueueStateTracker(CVMPTracker):
    detector_only = True
    def __init__(self, topology, graph):
        super().__init__(topology, graph)
        lanes = defaultdict(set)
        for m in topology.movements.values():
            for i in m.link_indices:
                lanes[topology.tls_link_from_lane[(m.tls_id, i)]].add(topology.pressure_movement(m.movement_id))
        self.lane_reps = {lane: tuple(sorted(reps)) for lane, reps in lanes.items()}

    def _update_vehicle(self, vehicle_id, road, now_s, vehicle_api):
        link = self.topology.edge_to_link.get(road)
        if link is not None:
            if self.current_link.get(vehicle_id) != link:
                self.link_entry_time[vehicle_id] = int(now_s)
            self.current_link[vehicle_id] = link
        reps = self.lane_reps.get(str(vehicle_api.getLaneID(vehicle_id)), ())
        if reps:
            self.current_movement[vehicle_id] = reps[0]
            self.current_is_bypass[vehicle_id] = False
        elif not road.startswith(':'):
            self.current_movement.pop(vehicle_id, None)
        self.last_road[vehicle_id] = road

    def build_state(self, now_s):
        raw = {mid: 0.0 for mid in self.topology.routing_movements}
        state = {mid: 0.0 for mid in self.topology.movements}
        by = defaultdict(list)
        for lane, reps in self.lane_reps.items():
            density = traci.lane.getLastStepVehicleNumber(lane) / traci.lane.getLength(lane)
            # Share an ambiguous lane across distinct movements, without routes
            # or duplicate counts; merged right/through groups count only once.
            for rep in reps:
                state[rep] += density / len(reps)
                raw[rep] += density / len(reps)
            for vehicle in traci.lane.getLastStepVehicleIDs(lane):
                rep = reps[0]
                link = self.topology.movements[rep].source_link
                by[rep].append(VehicleSnapshot(vehicle, link, rep,
                    self.link_entry_time.get(vehicle, int(now_s)), 0.0))
        self.last_raw_w = raw
        return state, by, 0

class HistoricalTurningEstimator(FixedTurningEstimator):
    """Use local observed exits, including uncontrolled exits in the denominator."""
    def __init__(self, topology, window_s):
        super().__init__(topology, window_s)
        self.previous_crossing = {}
        self.pending_terminal = {}
        self.previous_lane = {}
        self.lane_turns = defaultdict(deque)
        self.link_turns = defaultdict(deque)

    def observe_crossings(self, now_s, tracker, roads, vehicle_api, teleporting=(), lanes=None):
        lanes = lanes or {}
        for vehicle in teleporting:
            self.previous_crossing.pop(vehicle, None)
            self.pending_terminal.pop(vehicle, None)
            self.previous_lane.pop(vehicle, None)
        for vehicle in tracker.active_connected - set(teleporting):
            old, new = tracker.last_road.get(vehicle), roads.get(vehicle)
            old_lane = self.previous_lane.get(vehicle)
            if vehicle in lanes:
                self.previous_lane[vehicle] = lanes[vehicle]
            if not old or not new or old == new:
                continue
            if new.startswith(':'):
                if not old.startswith(':'):
                    self.pending_terminal[vehicle] = (old, old_lane)
                continue
            source, source_lane = self.pending_terminal.pop(vehicle, (old, old_lane))
            mid = self.topology.routing_movement_by_edge_pair.get((source, new))
            if mid is None:
                continue
            source_link = self.topology.routing_movements[mid].source_link
            self.link_turns[source_link].append((int(now_s), mid))
            if source_lane is not None:
                self.lane_turns[source_lane].append((int(now_s), mid))
            if mid not in self.topology.movements:
                # An early bypass is a local exit, not a later signalized movement.
                self.previous_crossing.pop(vehicle, None)
                continue
            prior = self.previous_crossing.get(vehicle)
            if prior is not None and mid in self.topology.routing_targets_by_source.get(prior, ()):
                self.history[prior].append((int(now_s), mid))
            self.previous_crossing[vehicle] = mid
        for vehicle in set(self.previous_crossing) - tracker.active_connected:
            self.previous_crossing.pop(vehicle, None)
            self.pending_terminal.pop(vehicle, None)
        for vehicle in set(self.previous_lane) - tracker.active_connected:
            self.previous_lane.pop(vehicle, None)
        cutoff = int(now_s) - self.window_s
        for events in (*self.history.values(), *self.lane_turns.values(), *self.link_turns.values()):
            while events and events[0][0] <= cutoff:
                events.popleft()

    def row(self, source_movement):
        movement = self.topology.movements.get(source_movement)
        if movement is None:
            return {}
        link_id = self.topology.edge_to_link.get(movement.to_edge)
        if link_id is None:
            return {}
        targets = self.topology.routing_movements_by_source_link.get(link_id, ())
        if not targets:
            return {}
        counts = defaultdict(int)
        for _, target in self.link_turns.get(link_id, ()):
            if target in targets:
                counts[target] += 1
        total = sum(counts.values())
        # With no local samples, use equal shares over ALL exits, including bypasses.
        shares = ({target: counts[target] / total for target in targets} if total
                  else {target: 1.0 / len(targets) for target in targets})
        return {target: share for target, share in shares.items()
                if target in self.topology.movements}

    def local_shares(self, link_id, lane_id, allowed_movements):
        allowed = tuple(allowed_movements)
        if not allowed:
            return {}
        for events in (self.lane_turns.get(lane_id, ()), self.link_turns.get(link_id, ())):
            counts = defaultdict(int)
            for _, movement_id in events:
                if movement_id in allowed:
                    counts[movement_id] += 1
            total = sum(counts.values())
            if total:
                return {movement_id: counts[movement_id] / total for movement_id in allowed}
        share = 1.0 / len(allowed)
        return {movement_id: share for movement_id in allowed}

class QueueMPController(CVMPController):
    def __init__(self, topology, config, turning, graph):
        super().__init__(topology, config, turning)
        self.graph = graph
