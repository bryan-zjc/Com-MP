"""Detector lane-density state with CV-MP topology and service semantics."""
from collections import defaultdict
import xml.etree.ElementTree as ET
import traci
from .env_CVMP import CVMPController, CVMPTracker


def _bus_only(allowed):
    allowed = set(allowed)
    return "bus" in allowed and "passenger" not in allowed


class QueueMPTracker(CVMPTracker):
    def __init__(self, topology, graph):
        super().__init__(topology, graph)
        self._pressure_lanes = defaultdict(set)
        for mid, movement in topology.movements.items():
            rep = topology.pressure_movement(mid)
            for index in movement.link_indices:
                self._pressure_lanes[rep].add(
                    topology.tls_link_from_lane[(movement.tls_id, index)])
        self._lane_lengths = {}

    def build_state(self, now_s):
        # Route snapshots serve the existing flow estimator/failure monitor,
        # never the detector-supported pressure state.
        _, by, unknown = super().build_state(now_s)
        lanes = {lane for group in self._pressure_lanes.values() for lane in group}
        density = {}
        for lane in lanes:
            if lane not in self._lane_lengths:
                self._lane_lengths[lane] = traci.lane.getLength(lane)
            length = self._lane_lengths[lane]
            if length <= 0:
                raise ValueError(f'Nonpositive lane length: {lane}')
            density[lane] = traci.lane.getLastStepVehicleNumber(lane) / length
        state = {mid: 0.0 for mid in self.topology.routing_movements}
        state.update({rep: sum(density[lane] for lane in group)
                      for rep, group in self._pressure_lanes.items()})
        # The inherited group transition mixing uses counts, not travel times.
        self.last_raw_w = {mid: float(len(by.get(mid, ())))
                           for mid in self.topology.routing_movements}
        return state, by, unknown


class FullLinkQueueMPTracker(QueueMPTracker):
    """Detector counts across each complete logical link, allocated without routes."""

    def __init__(self, topology, graph, network_path, turning):
        super().__init__(topology, graph)
        self.turning = turning
        root = ET.parse(network_path).getroot()
        lanes_by_edge = {
            edge.attrib["id"]: tuple(
                lane.attrib["id"] for lane in edge.findall("lane")
                if not _bus_only(lane.get("allow", "").split()))
            for edge in root.findall("edge")
        }
        lane_connections = defaultdict(set)
        for connection in root.findall("connection"):
            source = connection.attrib.get("from")
            target = connection.attrib.get("to")
            from_lane = connection.attrib.get("fromLane")
            to_lane = connection.attrib.get("toLane")
            if source is not None and target is not None and from_lane is not None and to_lane is not None:
                lane_connections[f"{source}_{from_lane}"].add(f"{target}_{to_lane}")

        self._full_link_lanes = {}
        self._lane_movements = {}
        for link_id, link in topology.links.items():
            link_lanes = tuple(lane for edge in link.edges for lane in lanes_by_edge[edge])
            link_lane_set = set(link_lanes)
            terminal_members = defaultdict(set)
            for movement_id in link.movement_ids:
                movement = topology.movements[movement_id]
                for index in movement.link_indices:
                    lane = topology.tls_link_from_lane[(movement.tls_id, index)]
                    terminal_members[lane].add(movement_id)

            reachable = {}

            def destinations(lane, visiting=frozenset()):
                if lane in reachable:
                    return reachable[lane]
                if lane in visiting:
                    return set()
                result = set(terminal_members.get(lane, ()))
                for downstream in lane_connections.get(lane, ()):
                    if downstream in link_lane_set:
                        result.update(destinations(downstream, visiting | {lane}))
                reachable[lane] = result
                return result

            self._full_link_lanes[link_id] = link_lanes
            straight = tuple(mid for mid in link.movement_ids
                             if "s" in topology.movements[mid].directions)
            for lane in link_lanes:
                allowed = destinations(lane)
                self._lane_movements[lane] = tuple(sorted(allowed or straight or link.movement_ids))

    def build_state(self, now_s):
        # Retain snapshots for the existing service-failure monitor; they do not
        # determine detector pressure or local turn allocations.
        _, by, unknown = CVMPTracker.build_state(self, now_s)
        state = {mid: 0.0 for mid in self.topology.routing_movements}
        raw = {mid: 0.0 for mid in self.topology.routing_movements}
        for link_id, lanes in self._full_link_lanes.items():
            length = self.topology.links[link_id].length_m
            if length <= 0:
                raise ValueError(f"Nonpositive logical link length: {link_id}")
            for lane in lanes:
                count = traci.lane.getLastStepVehicleNumber(lane)
                if count <= 0:
                    continue
                shares = self.turning.local_shares(
                    link_id, lane, self._lane_movements[lane])
                for movement_id, fraction in shares.items():
                    allocated = count * fraction
                    raw[movement_id] += allocated
                    state[self.topology.pressure_movement(movement_id)] += allocated / length
        self.last_raw_w = raw
        return state, by, unknown


class FullServiceQueueMPController(CVMPController):
    """Q-MP movement service is not halved for a physically shared lane."""

    def __init__(self, topology, config, turning):
        super().__init__(topology, config, turning)
        self._bus_lane_cache = {}
        self._receiving_lane_cache = {}
        for (tls_id, phase_id), reps in self._phase_representatives.items():
            signal = topology.phase_states[tls_id][phase_id]
            for rep in reps:
                movement = topology.movements[rep]
                lanes = {
                    topology.tls_link_from_lane[(tls_id, index)]
                    for index in movement.link_indices
                    if index < len(signal) and signal[index] in "gG"
                    and not self._is_bus_lane(
                        topology.tls_link_from_lane[(tls_id, index)])
                }
                self._phase_physical_lane_weight[(tls_id, phase_id, rep)] = float(len(lanes))

    def _is_bus_lane(self, lane):
        if lane not in self._bus_lane_cache:
            self._bus_lane_cache[lane] = _bus_only(traci.lane.getAllowed(lane))
        return self._bus_lane_cache[lane]

    def _service_lane_allowed(self, lane):
        return not self._is_bus_lane(lane)

    def _group_fixed_row(self, rep, raw_w):
        members = self._group_members(rep)
        weights = {m: max(float(raw_w.get(m, 0.0)), 0.0) for m in members}
        total = sum(weights.values())
        if total <= 1e-12:
            weights = {m: 1.0 for m in members}
            total = float(len(members))
        row = defaultdict(float)
        for member in members:
            for target, share in self.turning.row(member).items():
                row[target] += weights[member] / total * share
        # Do not renormalize: omitted uncontrolled exits have zero pressure.
        return dict(row)

    def service_capacity_audit(self):
        rows = super().service_capacity_audit()
        for row in rows:
            row["active_source_lanes"] = " ".join(
                lane for lane in row["active_source_lanes"].split()
                if not self._is_bus_lane(lane))
        return rows

    def service_amount(self, movement, phase_id, service_factor):
        weight = self._serviceable_lane_weight(movement, phase_id)
        if weight <= 0.0:
            return 0.0
        edge = movement.to_edge
        if edge not in self._receiving_lane_cache:
            all_lanes = tuple(f"{edge}_{i}" for i in range(traci.edge.getLaneNumber(edge)))
            lanes = tuple(lane for lane in all_lanes if not self._is_bus_lane(lane))
            self._receiving_lane_cache[edge] = (lanes, len(lanes) != len(all_lanes))
        lanes, excludes_bus = self._receiving_lane_cache[edge]
        if not excludes_bus:
            return super().service_amount(movement, phase_id, service_factor)
        if not lanes:
            return 0.0
        if edge not in self._receiving_slots:
            if edge not in self._receiving_capacity:
                self._receiving_capacity[edge] = sum(
                    traci.lane.getLength(lane) / 7.5 for lane in lanes)
            capacity = self._receiving_capacity[edge]
            count = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)
            occupied = sum(traci.lane.getLength(lane) / 7.5
                           * traci.lane.getLastStepOccupancy(lane) / 100.0
                           for lane in lanes)
            self._receiving_slots[edge] = max(0.0, capacity - max(count, occupied))
        nominal = (self.config.saturation_flow_vph_per_lane / 3600.0 * weight
                   * self.config.control_interval_s * service_factor)
        return min(nominal, self._receiving_slots[edge])
