"""MAPPO environment composing the current Q-MP and CV-MP evaluations."""

from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from src.road_env.sumo_bootstrap import ensure_sumo_python_tools

ensure_sumo_python_tools()
import traci
import traci.constants as tc

from src.road_env.config import CVMPConfig
from src.road_env.env_CVMP import (
    CVMPController, CVMPTracker, ConnectedServiceFailureMonitor,
    FixedTurningEstimator, SimpleSignalCoordinator,
)
from src.road_env.qmp_topology import (
    HistoricalTurningEstimator,
)
from src.road_env.env_MP import FullLinkQueueMPTracker, FullServiceQueueMPController
from src.road_env.runner import _aggregate_counts, _is_connected, _sumo_command
from src.road_env.topology import load_topology


class DetectorSlimHybridMAPPOEnv:
    PHASE_SLOTS = 4
    OBSERVATION_SIZE = 70
    OBSERVATION_SCHEMA = "v9_movement_queue_phase_context"
    MOVEMENT_SLOTS = (
        ("E", "s"), ("E", "l"), ("S", "s"), ("S", "l"),
        ("W", "s"), ("W", "l"), ("N", "s"), ("N", "l"),
    )

    def __init__(self, route_file, net_file, max_steps, seed, *,
                 cv_penetration_rate, alpha_bins,
                 queue_reduction_reward_weight=1.0,
                 queue_level_reward_weight=0.10, reward_scale=1.0,
                 gamma_per_second=0.997, decision_interval_seconds=15,
                 route_defined_cv=False, stop_exactly_at_max_steps=True,
                 demand_end_s=None, clear_when_empty=False):
        self.route_file = Path(route_file).resolve()
        self.net_file = Path(net_file).resolve()
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        if int(decision_interval_seconds) != 15:
            raise ValueError('Com-MP requires a 15-second decision interval')
        self.alpha_bins = np.asarray(alpha_bins, dtype=np.float32)
        self.reward_scale = float(reward_scale)
        self.queue_reduction_reward_weight = float(queue_reduction_reward_weight)
        self.queue_level_reward_weight = float(queue_level_reward_weight)
        if self.queue_reduction_reward_weight <= 0 or self.queue_level_reward_weight <= 0:
            raise ValueError("Both queue reward weights must be positive")
        self.gamma_per_second = float(gamma_per_second)
        self.route_defined_cv = bool(route_defined_cv)
        self.stop_exactly_at_max_steps = bool(stop_exactly_at_max_steps)
        self.clear_when_empty = bool(clear_when_empty)
        self.demand_end_s = int(demand_end_s or max_steps)
        self.config = CVMPConfig(
            sumo_seed=self.seed, connected_penetration=float(cv_penetration_rate),
            control_interval_s=int(decision_interval_seconds), yellow_s=3,
            maximum_red_s=180, turning_window_s=300,
            maximum_simulation_s=self.max_steps,
            demand_end_s=self.demand_end_s,
            show_progress=False,
        )
        self.config.validate()
        topology_file = self.net_file.parent / "topology.json"
        self.topology, self.graph = load_topology(topology_file, self.net_file)
        self.tls_ids = tuple(sorted(self.topology.phase_states))
        if any(len(states) > self.PHASE_SLOTS for states in self.topology.phase_states.values()):
            raise ValueError(
                "The observation template supports at most four candidate phases. "
                "Extend the common phase template and retrain before using this network.")
        self.max_actions = max(4, max(map(len, self.topology.phase_states.values())))
        self.net_info = {
            tls: {"phase_num": 2 * len(self.topology.phase_states[tls])}
            for tls in self.tls_ids
        }
        self.active = False
        self.reward_active = False
        self.now = 0

    def reset(self, tripinfo_path):
        self.tripinfo_path = Path(tripinfo_path).resolve()
        self.tripinfo_path.parent.mkdir(parents=True, exist_ok=True)
        paths = {"network": self.net_file, "route": self.route_file}
        command = _sumo_command(self.config, paths, self.tripinfo_path.parent)
        command[command.index("--tripinfo-output") + 1] = str(self.tripinfo_path)
        traci.start(command, numRetries=3)
        self.active = True
        self.now = 0
        self.reward_active = False
        self.q_turning = HistoricalTurningEstimator(self.topology, 300)
        self.q_tracker = FullLinkQueueMPTracker(
            self.topology, self.graph, self.net_file, self.q_turning)
        self.cv_tracker = CVMPTracker(self.topology, self.graph)
        self.cv_turning = FixedTurningEstimator(
            self.topology, self.config.turning_window_s)
        self.q_controller = FullServiceQueueMPController(
            self.topology, self.config, self.q_turning)
        self.cv_controller = CVMPController(
            self.topology, self.config, self.cv_turning)
        self.monitor = ConnectedServiceFailureMonitor(self.topology, self.config)
        self.q_monitor = ConnectedServiceFailureMonitor(self.topology, self.config)
        self.coordinators = {
            tls: SimpleSignalCoordinator(
                states, self.config.yellow_s, self.config.control_interval_s,
                self.config.maximum_red_s)
            for tls, states in self.topology.phase_states.items()
        }
        self.schedules = {
            tls: deque([states[0]] * self.config.control_interval_s)
            for tls, states in self.topology.phase_states.items()
        }
        self.last_signal = {}
        self.phase_start_time = {tls: 0 for tls in self.tls_ids}
        for tls, states in self.topology.phase_states.items():
            traci.trafficlight.setRedYellowGreenState(tls, states[0])
            self.last_signal[tls] = states[0]
        self.monitor.initialize_interval({
            tls: 0 for tls in self.tls_ids
        })
        self.q_monitor.initialize_interval({tls: 0 for tls in self.tls_ids})
        self.cv_ids = set()
        self.latest_speeds = {}
        self.last_alpha = {tls: 0.5 for tls in self.tls_ids}
        self.detector_available = {tls: True for tls in self.tls_ids}
        self.interval_start_time = {}
        self.interval_start_queues = {}
        self._pressure_cache_time = -1
        self._build_lane_groups()
        self._build_phase_context()
        self._build_reward_lanes()
        flags = {tls: 0 for tls in self.tls_ids}
        return self.get_all_observations(), self.net_info, flags

    @staticmethod
    def _approach_direction(lane_id):
        shape = traci.lane.getShape(lane_id)
        if len(shape) < 2:
            return None
        dx = shape[0][0] - shape[-1][0]
        dy = shape[0][1] - shape[-1][1]
        if abs(dx) >= abs(dy):
            return "E" if dx > 0 else "W"
        return "N" if dy > 0 else "S"

    def _build_lane_groups(self):
        self.movement_lanes = {}
        self.slot_movements = {}
        self.observation_links = {}
        self.incoming_lanes = {}
        self.observation_lanes = set()
        for tls, states in self.topology.phase_states.items():
            groups = {slot: set() for slot in self.MOVEMENT_SLOTS}
            slot_movements = {slot: set() for slot in self.MOVEMENT_SLOTS}
            slot_by_movement = {}
            all_lanes = set()
            for mid, movement in self.topology.movements.items():
                if movement.tls_id != tls or self.topology.pressure_movement(mid) != mid:
                    continue
                turn = "l" if "l" in movement.directions else "s"
                for index in movement.link_indices:
                    lane = self.topology.tls_link_from_lane.get((tls, int(index)))
                    if lane is None:
                        continue
                    all_lanes.add(lane)
                    slot = (self._approach_direction(lane), turn)
                    if slot in groups:
                        groups[slot].add(lane)
                        slot_movements[slot].add(mid)
                        slot_by_movement[mid] = slot
            self.movement_lanes[tls] = groups
            self.slot_movements[tls] = slot_movements
            self.incoming_lanes[tls] = all_lanes
            observation_links = []
            full_lanes = set()
            for link_id, link in self.topology.links.items():
                if link.downstream_tls != tls:
                    continue
                link_lanes = self.q_tracker._full_link_lanes[link_id]
                observation_links.append((link_id, link_lanes))
                full_lanes.update(link_lanes)
            self.observation_links[tls] = (observation_links, slot_by_movement)
            self.observation_lanes.update(full_lanes)
            for lane in full_lanes:
                traci.lane.subscribe(lane, (
                    tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                    tc.LAST_STEP_VEHICLE_ID_LIST,
                ))

    def _build_phase_context(self):
        self.phase_movement_matrices = {}
        self.phase_valid_masks = {}
        self.phase_observation_order = {}
        for tls, states in self.topology.phase_states.items():
            count = len(states)
            if not 0 < count <= self.PHASE_SLOTS:
                raise ValueError(f"Unsupported candidate phase count for {tls}: {count}")
            rows = []
            for phase in range(count):
                served = self.q_controller._phase_representatives[(tls, phase)]
                rows.append(tuple(int(any(mid in served for mid in self.slot_movements[tls][slot]))
                                  for slot in self.MOVEMENT_SLOTS))
            # Canonicalize observation rows only; actual signal phase IDs stay unchanged.
            order = tuple(sorted(range(count), key=lambda phase: rows[phase], reverse=True))
            matrix = np.zeros((self.PHASE_SLOTS, len(self.MOVEMENT_SLOTS)), dtype=np.float32)
            matrix[:count] = np.asarray([rows[phase] for phase in order], dtype=np.float32)
            valid = np.zeros(self.PHASE_SLOTS, dtype=np.float32)
            valid[:count] = 1.0
            self.phase_movement_matrices[tls] = matrix
            self.phase_valid_masks[tls] = valid
            self.phase_observation_order[tls] = order

    def get_phase_context(self, tls):
        return np.concatenate((self.phase_movement_matrices[tls].ravel(),
                               self.phase_valid_masks[tls]))

    def _movement_queue_counts(self, tls, subscriptions=None):
        # Allocate each physical lane once; right turns use their pressure representative.
        subscriptions = (traci.lane.getAllSubscriptionResults() or {}
                         if subscriptions is None else subscriptions)
        queued = defaultdict(float)
        links, slot_by_movement = self.observation_links[tls]
        seen_lanes = set()
        for link_id, lanes in links:
            for lane in lanes:
                if lane in seen_lanes:
                    raise ValueError(f"Lane {lane} occurs in multiple incoming links for {tls}")
                seen_lanes.add(lane)
                allowed = tuple(mid for mid in self.q_tracker._lane_movements[lane]
                                if self.topology.pressure_movement(mid) in slot_by_movement)
                if not allowed:
                    allowed = tuple(mid for mid in self.topology.links[link_id].movement_ids
                                    if self.topology.pressure_movement(mid) in slot_by_movement)
                if not allowed:
                    raise ValueError(f"No controlled movement for incoming lane {lane}")
                shares = self.q_turning.local_shares(link_id, lane, allowed)
                value = (subscriptions.get(lane) or {}).get(
                    tc.LAST_STEP_VEHICLE_HALTING_NUMBER)
                if value is None:
                    value = traci.lane.getLastStepHaltingNumber(lane)
                for mid, fraction in shares.items():
                    slot = slot_by_movement[self.topology.pressure_movement(mid)]
                    queued[slot] += float(value) * fraction
        return np.asarray([queued[slot] for slot in self.MOVEMENT_SLOTS],
                          dtype=np.float32)

    def _build_reward_lanes(self):
        self.reward_lanes = {tls: set() for tls in self.tls_ids}
        for link_id, link in self.topology.links.items():
            if link.downstream_tls in self.reward_lanes:
                self.reward_lanes[link.downstream_tls].update(
                    self.q_tracker._full_link_lanes[link_id])
        if any(not lanes for lanes in self.reward_lanes.values()):
            raise ValueError("Every controlled intersection needs incoming reward lanes")
        for lanes in self.reward_lanes.values():
            for lane in lanes:
                if lane not in self.observation_lanes:
                    traci.lane.subscribe(lane, (tc.LAST_STEP_VEHICLE_HALTING_NUMBER,))

    def _intersection_queue(self, tls, subscriptions=None):
        subscriptions = (traci.lane.getAllSubscriptionResults() or {}
                         if subscriptions is None else subscriptions)
        total = 0
        for lane in self.reward_lanes[tls]:
            value = (subscriptions.get(lane) or {}).get(
                tc.LAST_STEP_VEHICLE_HALTING_NUMBER)
            if value is None:
                value = traci.lane.getLastStepHaltingNumber(lane)
            total += int(value)
        return total

    @staticmethod
    def _normalize_pressure(values):
        array = np.asarray(values, dtype=np.float64)
        spread = float(array.max() - array.min())
        if spread <= 0.0:
            return np.zeros_like(array)
        return (array - array.min()) / spread

    def _prepare_decision(self):
        if self._pressure_cache_time == self.now:
            return
        self._pressure_cache_time = self.now
        self.q_state, self.q_by, _ = self.q_tracker.build_state(self.now)
        self.cv_state, self.cv_by, _ = self.cv_tracker.build_state(self.now)
        self.q_controller.begin_decision()
        self.cv_controller.begin_decision()
        snapshots = [s for group in self.cv_by.values() for s in group]
        self.cv_turning.observe(self.now, snapshots, traci.vehicle)
        zero_service, _ = self.monitor.finalize_interval_and_triggers()
        q_zero_service, _ = self.q_monitor.finalize_interval_and_triggers()
        self.q_monitor.cache_movements([s for group in self.q_by.values() for s in group])
        self.q_waiting = _aggregate_counts(self.topology, {
            mid: sum(self.latest_speeds.get(s.vehicle_id, float('inf')) <=
                     self.config.service_failure_waiting_speed_mps for s in group)
            for mid, group in self.q_by.items()})
        self.q_zero_service = q_zero_service
        self.monitor.cache_movements(snapshots)
        self.zero_service = zero_service
        waiting_raw = {
            mid: sum(self.latest_speeds.get(s.vehicle_id, float("inf"))
                     <= self.config.service_failure_waiting_speed_mps
                     for s in group)
            for mid, group in self.cv_by.items()
        }
        self.waiting_connected = _aggregate_counts(
            self.topology, waiting_raw)
        self.cv_at_tls = {tls: 0 for tls in self.tls_ids}
        for mid, group in self.cv_by.items():
            movement = self.topology.movements.get(mid)
            if movement is not None:
                self.cv_at_tls[movement.tls_id] += len(group)
        self.pressures = {}
        self.movement_pressures = {}
        for tls in self.tls_ids:
            current = self.coordinators[tls].current_phase
            _, q_eval = self.q_controller.select_phase(
                tls, current, self.q_state, self.q_tracker.last_raw_w, q_zero_service)
            _, cv_eval = self.cv_controller.select_phase(
                tls, current, self.cv_state, self.cv_tracker.last_raw_w,
                zero_service)
            self.pressures[tls] = (
                np.asarray([ev.pressure_score for ev in q_eval]),
                np.asarray([ev.pressure_score for ev in cv_eval]),
            )
            self.movement_pressures[tls] = (
                self._movement_pressures(tls, q_eval),
                self._movement_pressures(tls, cv_eval),
            )

    def _movement_pressures(self, tls, evaluations):
        by_movement = defaultdict(float)
        for evaluation in evaluations:
            for mid, detail in evaluation.movement_details.items():
                by_movement[mid] = max(by_movement[mid], float(detail["pressure"]))
        return np.asarray([
            sum(by_movement[mid] for mid in self.slot_movements[tls][slot])
            for slot in self.MOVEMENT_SLOTS
        ], dtype=np.float32)

    def _normalized_movement_pressures(self, tls, values):
        available = np.asarray([
            bool(self.slot_movements[tls][slot]) for slot in self.MOVEMENT_SLOTS
        ])
        normalized = np.zeros(len(self.MOVEMENT_SLOTS), dtype=np.float32)
        if available.any():
            normalized[available] = self._normalize_pressure(values[available])
        return normalized

    def set_detector_available(self, tls, available):
        if tls not in self.detector_available:
            raise KeyError(tls)
        self.detector_available[tls] = bool(available)

    def _effective_alpha(self, tls, requested):
        if not self.detector_available[tls] and self.cv_at_tls[tls] > 0:
            return 1.0
        if self.cv_at_tls[tls] == 0:
            return 0.0
        return float(np.clip(requested, 0.0, 1.0))

    def _phase_for_alpha(self, tls, alpha):
        q_raw, cv_raw = self.pressures[tls]
        if not self.detector_available[tls] and self.cv_at_tls[tls] == 0:
            selected = self.coordinators[tls].round_robin_next_phase()
            return self.coordinators[tls].enforce_max_red(selected, self.now)
        alpha = self._effective_alpha(tls, alpha)
        current = self.coordinators[tls].current_phase
        raw = (1.0 - alpha) * q_raw + alpha * cv_raw
        if np.max(raw) <= 0.0:
            selected = self.coordinators[tls].round_robin_next_phase()
        else:
            combined = ((1.0 - alpha) * self._normalize_pressure(q_raw)
                        + alpha * self._normalize_pressure(cv_raw))
            selected = int(max(range(len(combined)),
                               key=lambda i: (combined[i], i == current, -i)))
        return self.coordinators[tls].enforce_max_red(selected, self.now)

    def get_action_mask(self, tls):
        return np.ones(len(self.alpha_bins), dtype=bool)

    def get_action_diagnostics(self, tls):
        self._prepare_decision()
        q_raw, cv_raw = self.pressures[tls]
        phases = [self._phase_for_alpha(tls, float(alpha))
                  for alpha in self.alpha_bins]
        adaptive = self.detector_available[tls] and self.cv_at_tls[tls] > 0
        return {
            "actor_valid": adaptive and len(set(phases)) > 1,
            "candidate_phase_count": len(set(phases)),
            "mp_phase": int(np.argmax(self._normalize_pressure(q_raw))),
            "cvmp_phase": int(np.argmax(self._normalize_pressure(cv_raw))),
            "phase_by_action": phases,
        }

    def get_observation(self, tls):
        self._prepare_decision()
        q_raw, cv_raw = self.movement_pressures[tls]
        q = (self._normalized_movement_pressures(tls, q_raw)
             if self.detector_available[tls] else
             np.zeros(len(self.MOVEMENT_SLOTS), dtype=np.float32))
        cv = self._normalized_movement_pressures(tls, cv_raw)
        active = np.zeros(len(self.MOVEMENT_SLOTS), dtype=np.float32)
        current = self.coordinators[tls].current_phase
        served = self.q_controller._phase_representatives[(tls, current)]
        for i, slot in enumerate(self.MOVEMENT_SLOTS):
            active[i] = float(any(mid in served for mid in self.slot_movements[tls][slot]))
        elapsed = max(0, self.now - self.phase_start_time[tls])
        lane_results = traci.lane.getAllSubscriptionResults() or {}
        ids = set()
        for lane in self.incoming_lanes[tls]:
            ids.update((lane_results.get(lane) or {}).get(
                tc.LAST_STEP_VEHICLE_ID_LIST, ()))
        local_cv_ratio = len(ids & self.cv_ids) / len(ids) if ids else 0.0
        observation = np.concatenate((
            q, cv, self._movement_queue_counts(tls, lane_results), active,
            np.asarray([elapsed, local_cv_ratio]),
            self.get_phase_context(tls),
        )).astype(np.float32)
        if self.requires_topology_mask(tls):
            observation *= self.get_topology_mask(tls)
        return observation

    def requires_topology_mask(self, tls):
        return (len(self.topology.phase_states[tls]) != 4 or
                any(not self.slot_movements[tls][slot]
                    for slot in self.MOVEMENT_SLOTS))

    def get_topology_mask(self, tls):
        movements = np.asarray(
            [bool(self.slot_movements[tls][slot])
             for slot in self.MOVEMENT_SLOTS], dtype=np.float32)
        return np.concatenate((movements, movements, movements, movements,
                               np.ones(2, dtype=np.float32),
                               np.repeat(self.phase_valid_masks[tls], len(self.MOVEMENT_SLOTS)),
                               self.phase_valid_masks[tls]))

    def get_all_observations(self):
        return {tls: self.get_observation(tls) for tls in self.tls_ids}

    def get_observation_matrix(self, tls_ids):
        return np.stack([self.get_observation(tls) for tls in tls_ids])

    @staticmethod
    def get_global_context(observations):
        matrix = np.asarray(observations, dtype=np.float32)
        return np.concatenate((matrix.mean(axis=0), matrix.max(axis=0)))

    def begin_control(self):
        self.reward_active = True
        subscriptions = traci.lane.getAllSubscriptionResults() or {}
        for tls in self.tls_ids:
            self.reset_interval_reward(tls, subscriptions)

    def reset_interval_reward(self, tls, subscriptions=None, queue=None):
        self.interval_start_time[tls] = self.now
        self.interval_start_queues[tls] = (
            self._intersection_queue(tls, subscriptions)
            if queue is None else queue)

    def consume_interval_reward(self, tls):
        subscriptions = traci.lane.getAllSubscriptionResults() or {}
        current = self._intersection_queue(tls, subscriptions)
        reduction = float(self.interval_start_queues[tls] - current)
        level = float(current)
        reward = self.reward_scale * (
            self.queue_reduction_reward_weight * reduction
            - self.queue_level_reward_weight * level
        )
        duration = self.now - self.interval_start_time[tls]
        self.reset_interval_reward(tls, queue=current)
        return reward, duration, {"queue_reduction": reduction, "queue_level": level}

    def _advance_one_second(self):
        for tls, schedule in self.schedules.items():
            if not schedule:
                raise RuntimeError(f"Empty signal schedule: {tls} at {self.now}")
            signal = schedule[0]
            if signal != self.last_signal[tls]:
                traci.trafficlight.setRedYellowGreenState(tls, signal)
                self.last_signal[tls] = signal
        traci.simulationStep()
        self.now += 1
        for schedule in self.schedules.values():
            schedule.popleft()
        teleports = set(traci.simulation.getStartingTeleportIDList())
        departed = tuple(map(str, traci.simulation.getDepartedIDList()))
        arrived = tuple(map(str, traci.simulation.getArrivedIDList()))
        self.monitor.observe_arrivals_before_removal(self.cv_tracker, arrived)
        self.q_monitor.observe_arrivals_before_removal(self.q_tracker, arrived)
        self.q_tracker.remove_arrived(arrived)
        self.cv_tracker.remove_arrived(arrived)
        self.monitor.remove_vehicles(arrived)
        self.q_monitor.remove_vehicles(arrived)
        live = [v for v in departed if v not in arrived]
        self.q_tracker.add_departed(live, self.now, traci.vehicle)
        connected = []
        for vehicle_id in live:
            if (traci.vehicle.getTypeID(vehicle_id) == "CV"
                    if self.route_defined_cv else
                    _is_connected(vehicle_id, self.seed,
                                  self.config.connected_penetration)):
                connected.append(vehicle_id)
                self.cv_ids.add(vehicle_id)
        self.cv_tracker.add_departed(connected, self.now, traci.vehicle)
        for vehicle_id in live:
            variables = (tc.VAR_ROAD_ID, tc.VAR_SPEED, tc.VAR_LANE_ID,
                         tc.VAR_LANEPOSITION)
            traci.vehicle.subscribe(vehicle_id, variables)
        subscriptions = traci.vehicle.getAllSubscriptionResults() or {}
        q_roads = {}
        q_lanes = {}
        cv_roads = {}
        self.latest_speeds = {}
        for vehicle_id in self.q_tracker.active_connected:
            values = subscriptions.get(vehicle_id) or {}
            road = values.get(tc.VAR_ROAD_ID)
            if road is None:
                road = traci.vehicle.getRoadID(vehicle_id)
            q_roads[vehicle_id] = str(road)
            lane = values.get(tc.VAR_LANE_ID)
            if lane is None:
                lane = traci.vehicle.getLaneID(vehicle_id)
            q_lanes[vehicle_id] = str(lane)
            speed = values.get(tc.VAR_SPEED)
            if speed is None:
                speed = traci.vehicle.getSpeed(vehicle_id)
            self.latest_speeds[vehicle_id] = float(speed)
            if vehicle_id in self.cv_tracker.active_connected:
                cv_roads[vehicle_id] = str(road)
                speed = values.get(tc.VAR_SPEED)
                if speed is None:
                    speed = traci.vehicle.getSpeed(vehicle_id)
                self.latest_speeds[vehicle_id] = float(speed)
        self.q_turning.observe_crossings(
            self.now, self.q_tracker, q_roads, traci.vehicle, teleports,
            q_lanes)
        self.monitor.observe_road_transitions(
            self.cv_tracker, cv_roads, traci.vehicle)
        self.q_monitor.observe_road_transitions(self.q_tracker, q_roads, traci.vehicle)
        self.q_tracker.update_roads(self.now, q_roads, traci.vehicle)
        self.cv_tracker.update_roads(self.now, cv_roads, traci.vehicle)
        self.cv_ids.difference_update(arrived)
        self._pressure_cache_time = -1

    def fixed_time_warm_start(self, steps):
        if steps:
            raise ValueError('Warm-start is disabled in v5')
        limit = min(self.max_steps, self.now + int(steps))
        while self.now < limit:
            if self.now > 0 and self.now % self.config.control_interval_s == 0:
                chosen = {}
                for tls, coordinator in self.coordinators.items():
                    phase = coordinator.round_robin_next_phase()
                    self.schedules[tls] = deque(coordinator.schedule(phase, self.now))
                    self.phase_start_time[tls] = self.now + self.config.yellow_s
                    chosen[tls] = phase
                self.monitor.initialize_interval(chosen)
            self._advance_one_second()
        flags = {tls: int(self.now > 0 and
                          self.now % self.config.control_interval_s == 0)
                 for tls in self.tls_ids}
        return {}, flags

    def step(self, alpha_action):
        if self.now > 0 and self.now % self.config.control_interval_s == 0:
            self._prepare_decision()
            selected = {}
            for tls, coordinator in self.coordinators.items():
                alpha = float(alpha_action.get(tls, self.last_alpha[tls]))
                self.last_alpha[tls] = alpha
                phase = self._phase_for_alpha(tls, alpha)
                if phase != coordinator.current_phase:
                    self.phase_start_time[tls] = self.now + self.config.yellow_s
                selected[tls] = phase
                self.schedules[tls] = deque(coordinator.schedule(phase, self.now))
            self.monitor.initialize_interval(
                selected, waiting_connected=self.waiting_connected,
                zero_service=self.zero_service)
            self.q_monitor.initialize_interval(
                selected, waiting_connected=self.q_waiting,
                zero_service=self.q_zero_service)
        self._advance_one_second()
        done = self.now >= self.max_steps or (
            self.clear_when_empty and self.now >= self.demand_end_s and
            traci.simulation.getMinExpectedNumber() <= 0)
        flags = {tls: int(not done and self.now > 0 and
                          self.now % self.config.control_interval_s == 0)
                 for tls in self.tls_ids}
        return {}, {tls: [done] for tls in self.tls_ids}, flags, {}

    def close(self):
        if self.active:
            traci.close()
            self.active = False
