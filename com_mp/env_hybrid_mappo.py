from collections import defaultdict

import numpy as np
import traci
import traci.constants as tc

from src.road_env.env_hybrid_MP import HybridMPEnv


class HybridMAPPOEnv(HybridMPEnv):
    """Hybrid-MP environment with subscription caches and SMDP rewards."""

    def __init__(self, *args, alpha_bins, queue_reward_weight=1.0,
                 waiting_reward_weight=1.0, throughput_reward_weight=1.0,
                 switch_penalty=1.0, waiting_time_divisor=10.0,
                 reward_scale=0.01,
                 state_waiting_scale=300.0,
                 gamma_per_second=0.997, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_bins = np.asarray(alpha_bins, dtype=np.float32)
        self.queue_reward_weight = float(queue_reward_weight)
        self.waiting_reward_weight = float(waiting_reward_weight)
        self.throughput_reward_weight = float(throughput_reward_weight)
        self.switch_penalty = float(switch_penalty)
        self.waiting_time_divisor = float(waiting_time_divisor)
        self.reward_scale = float(reward_scale)
        self.hold_current_on_cvmp_tie = False
        self.track_cv_state = True
        self.track_waiting_state = True
        self.state_waiting_scale = float(state_waiting_scale)
        self.gamma_per_second = float(gamma_per_second)
        self.runtime_cache_ready = False
        self.lane_cache = {}
        self.lane_capacity = {}
        self.lane_owner = {}
        self.lane_waiting_sum = defaultdict(float)
        self.lane_max_waiting = defaultdict(float)
        self.previous_tls_vehicle_ids = defaultdict(set)
        self.tls_departures_step = defaultdict(int)
        self.pending_switch_events = defaultdict(int)
        self.interval_reward = defaultdict(float)
        self.interval_components = defaultdict(lambda: defaultdict(float))
        self.interval_discount = defaultdict(lambda: 1.0)
        self.interval_duration = defaultdict(int)
        self.reward_active = False
        self.max_actions = 0
        self.spillover_tracking_active = False
        self.spillover_count_history = []

    def reset(self, tripinfo_path):
        self.runtime_cache_ready = False
        self.lane_cache = {}
        self.lane_capacity = {}
        self.lane_owner = {}
        self.lane_waiting_sum = defaultdict(float)
        self.lane_max_waiting = defaultdict(float)
        self.previous_tls_vehicle_ids = defaultdict(set)
        self.tls_departures_step = defaultdict(int)
        self.pending_switch_events = defaultdict(int)
        self.interval_reward = defaultdict(float)
        self.interval_components = defaultdict(lambda: defaultdict(float))
        self.interval_discount = defaultdict(lambda: 1.0)
        self.interval_duration = defaultdict(int)
        self.reward_active = False
        self.spillover_tracking_active = False
        self.spillover_count_history = []
        states, net_info, decision_flag = super().reset(tripinfo_path)
        self._ensure_runtime_cache()
        self.max_actions = max(info["phase_num"] // 2 for info in self.net_info.values())
        return self.get_all_observations(), net_info, decision_flag

    def begin_control(self):
        self.reward_active = True
        self.spillover_tracking_active = True
        self._record_source_spillover()
        for tls_id in self.net_info:
            self.reset_interval_reward(tls_id)

    def _record_source_spillover(self):
        if not self.spillover_tracking_active:
            return
        self.spillover_count_history.append((
            float(traci.simulation.getTime()),
            len(traci.simulation.getPendingVehicles()),
        ))

    def get_source_spillover_summary(self, slope_window_seconds=300.0):
        if not self.spillover_count_history:
            return {
                "max_source_spillover_count": 0,
                "time_of_max_source_spillover_s": 0.0,
                "mean_source_spillover_count": 0.0,
                "final_source_spillover_count": 0,
                "spillover_vehicle_seconds": 0.0,
                "spillover_nonzero_fraction": 0.0,
                "spillover_last_300s_slope_veh_per_s": 0.0,
                "spillover_observation_steps": 0,
            }

        history = np.asarray(self.spillover_count_history, dtype=float)
        times = history[:, 0]
        counts = history[:, 1]
        max_index = int(np.argmax(counts))
        if len(times) > 1:
            trapezoid = getattr(np, "trapezoid", np.trapz)
            vehicle_seconds = float(trapezoid(counts, times))
        else:
            vehicle_seconds = 0.0
        window_start = max(times[0], times[-1] - slope_window_seconds)
        window_mask = times >= window_start
        if np.count_nonzero(window_mask) >= 2:
            window_times = times[window_mask]
            window_counts = counts[window_mask]
            centered_times = window_times - float(np.mean(window_times))
            denominator = float(np.sum(centered_times * centered_times))
            slope = (
                float(np.sum(
                    centered_times *
                    (window_counts - float(np.mean(window_counts))))) /
                denominator
                if denominator > 0.0 else 0.0
            )
        else:
            slope = 0.0
        return {
            "max_source_spillover_count": int(counts[max_index]),
            "time_of_max_source_spillover_s": float(times[max_index]),
            "mean_source_spillover_count": float(np.mean(counts)),
            "final_source_spillover_count": int(counts[-1]),
            "spillover_vehicle_seconds": vehicle_seconds,
            "spillover_nonzero_fraction": float(np.mean(counts > 0)),
            "spillover_last_300s_slope_veh_per_s": slope,
            "spillover_observation_steps": int(len(counts)),
        }

    def _ensure_runtime_cache(self):
        if self.runtime_cache_ready or not self.net_info:
            return
        tracked_lanes = set()
        for tls_id, info in self.net_info.items():
            for lane_id in info.get("lanes_in", ()):
                tracked_lanes.add(lane_id)
                self.lane_owner[lane_id] = tls_id
        for lane_id in tracked_lanes:
            traci.lane.subscribe(lane_id, (tc.LAST_STEP_VEHICLE_NUMBER,
                                           tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                                           tc.LAST_STEP_VEHICLE_ID_LIST))
            length = max(traci.lane.getLength(lane_id), 7.5)
            self.lane_capacity[lane_id] = max(1.0, length / 7.5)
        self.runtime_cache_ready = True
        self._refresh_lane_cache()

    def _refresh_lane_cache(self):
        self.lane_cache = traci.lane.getAllSubscriptionResults() or {}

    def _lane_value(self, lane_id, variable, default=0):
        lane_result = self.lane_cache.get(lane_id)
        if lane_result is not None and variable in lane_result:
            return lane_result[variable]
        if variable == tc.LAST_STEP_VEHICLE_NUMBER:
            return traci.lane.getLastStepVehicleNumber(lane_id)
        if variable == tc.LAST_STEP_VEHICLE_HALTING_NUMBER:
            return traci.lane.getLastStepHaltingNumber(lane_id)
        if variable == tc.LAST_STEP_VEHICLE_ID_LIST:
            return traci.lane.getLastStepVehicleIDs(lane_id)
        return default

    def _collect_mp_movement_states(self):
        """Use lane subscription results instead of one TraCI call per movement."""
        self._ensure_mp_network_info()
        current_time = traci.simulation.getTime()
        if self.mp_state_cache_time == current_time:
            return self.mp_movement_states, self.mp_movement_counts
        self.mp_state_cache_time = current_time
        self.mp_movement_states = defaultdict(float)
        self.mp_movement_counts = defaultdict(int)
        for movement in self.mp_all_movements:
            count = int(self._lane_value(movement["in_lane"],
                                         tc.LAST_STEP_VEHICLE_NUMBER))
            length = self._physical_length(movement["in_lane"])
            self.mp_movement_states[movement["key"]] = count / length
            self.mp_movement_counts[movement["key"]] = count
        return self.mp_movement_states, self.mp_movement_counts

    def _update_cv_vehicle_states(self):
        if not self.track_cv_state and not self.track_waiting_state:
            self.lane_waiting_sum = defaultdict(float)
            self.lane_max_waiting = defaultdict(float)
            return

        self._ensure_cvmp_network_info()
        current_time = traci.simulation.getTime()
        if self.track_cv_state:
            for veh_id in traci.simulation.getDepartedIDList():
                if self._is_cv_vehicle(veh_id):
                    self._cache_vehicle_route(veh_id)
            for veh_id in traci.simulation.getArrivedIDList():
                self._remove_cv_vehicle(veh_id)

        # Only vehicles on controlled entrance lanes can contribute to CV-MP.
        # Lane subscriptions provide both the vehicle set and its lane without
        # iterating over every active CV in the full network.
        relevant = {}
        tracked_waiting = set()
        current_tls_vehicle_ids = defaultdict(set)
        self.lane_waiting_sum = defaultdict(float)
        self.lane_max_waiting = defaultdict(float)
        for lane_id in self.lane_capacity:
            vehicle_ids = self._lane_value(
                lane_id, tc.LAST_STEP_VEHICLE_ID_LIST, ())
            tls_id = self.lane_owner[lane_id]
            track_throughput = lane_id in self.net_info[tls_id]["lanes_in_no_right"]
            for veh_id in vehicle_ids:
                if self.track_waiting_state:
                    tracked_waiting.add(veh_id)
                    if self.vehicle_wait_lane.get(veh_id) != lane_id:
                        self.vehicle_wait_lane[veh_id] = lane_id
                        self.vehicle_lane_waiting_time[veh_id] = 0.0
                    if traci.vehicle.getSpeed(veh_id) <= self.waiting_speed_threshold:
                        self.vehicle_lane_waiting_time[veh_id] += 1.0
                        wait = self.vehicle_lane_waiting_time[veh_id]
                        self.lane_waiting_sum[lane_id] += wait
                        self.lane_max_waiting[lane_id] = max(
                            self.lane_max_waiting[lane_id], wait)
                    if track_throughput:
                        current_tls_vehicle_ids[tls_id].add(veh_id)
                if self.track_cv_state and veh_id in self.cv_vehicle_ids:
                    relevant[veh_id] = lane_id

        if self.track_waiting_state:
            for veh_id in list(self.vehicle_wait_lane):
                if veh_id not in tracked_waiting:
                    self.vehicle_wait_lane.pop(veh_id, None)
                    self.vehicle_lane_waiting_time.pop(veh_id, None)

            for tls_id in self.net_info:
                previous = self.previous_tls_vehicle_ids[tls_id]
                current = current_tls_vehicle_ids[tls_id]
                self.tls_departures_step[tls_id] = len(previous.difference(current))
                self.previous_tls_vehicle_ids[tls_id] = current

        if self.track_cv_state:
            for veh_id in self.active_cv_ids.difference(relevant):
                self._remove_vehicle_contribution(veh_id)
                self.vehicle_lane.pop(veh_id, None)
                self.vehicle_edge.pop(veh_id, None)
                self.vehicle_edge_enter_time.pop(veh_id, None)

            self.active_cv_ids = set(relevant)
            for veh_id, lane_id in relevant.items():
                if veh_id not in self.vehicle_route:
                    self._cache_vehicle_route(veh_id)
                self._update_cv_vehicle_movement(veh_id, lane_id, current_time)

    def _collect_hybrid_pressure(self, tls_id, alpha):
        if alpha <= 0.0:
            return self._normalize_pressure(self._collect_mp_pressure(tls_id))
        if alpha >= 1.0:
            return self._normalize_pressure(self._collect_cvmp_pressure(tls_id))
        return super()._collect_hybrid_pressure(tls_id, alpha)

    def _simulate(self, steps_todo=1):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step
        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            self._ensure_runtime_cache()
            self._refresh_lane_cache()
            self._update_cv_vehicle_states()
            self._record_source_spillover()
            if self.reward_active:
                self._accumulate_reward_step()
        return self._step

    def _set_control_phase(self, tls_id):
        """libsumo requires an integer phase index (socket TraCI coerces it)."""
        current_phase = traci.trafficlight.getPhase(tls_id)
        next_phase = int(self.control_input[tls_id] * 2)
        traci.trafficlight.setPhase(tls_id, next_phase)
        if current_phase != next_phase:
            self.phase_switch_pointer[tls_id] = traci.simulation.getTime()

    def _select_phase_by_alpha(self, tls_id, alpha):
        """Select the maximum pressure without hysteresis."""
        pressure = np.asarray(
            self._collect_hybrid_pressure(tls_id, alpha), dtype=np.float64)
        if self.hold_current_on_cvmp_tie and alpha >= 1.0:
            max_pressure = float(np.max(pressure))
            candidates = np.flatnonzero(np.isclose(
                pressure, max_pressure, rtol=0.0, atol=1e-9))
            current_action = int(
                traci.trafficlight.getPhase(tls_id) // 2)
            if current_action in candidates:
                return current_action
        return int(np.argmax(pressure))

    def _intersection_queue_ratio(self, tls_id):
        lanes = self.net_info[tls_id]["lanes_in_no_right"]
        queued = sum(float(self._lane_value(lane, tc.LAST_STEP_VEHICLE_HALTING_NUMBER))
                     for lane in lanes)
        capacity = sum(self.lane_capacity.get(lane, 1.0) for lane in lanes)
        return queued / max(capacity, 1.0)

    def _intersection_reward_terms(self, tls_id):
        lanes = self.net_info[tls_id]["lanes_in_no_right"]
        queued = sum(float(self._lane_value(
            lane, tc.LAST_STEP_VEHICLE_HALTING_NUMBER)) for lane in lanes)
        waiting = sum(self.lane_waiting_sum[lane] for lane in lanes)
        lane_count = max(len(lanes), 1)
        return {
            "queue": queued / lane_count,
            "waiting": waiting / lane_count / max(self.waiting_time_divisor, 1e-6),
            "throughput": self.tls_departures_step[tls_id] / lane_count,
            "switch": float(self.pending_switch_events.pop(tls_id, 0)),
        }

    def _accumulate_reward_step(self):
        for tls_id in self.net_info:
            terms = self._intersection_reward_terms(tls_id)
            raw_reward = (
                self.throughput_reward_weight * terms["throughput"] -
                self.queue_reward_weight * terms["queue"] -
                self.waiting_reward_weight * terms["waiting"] -
                self.switch_penalty * terms["switch"])
            discount = self.interval_discount[tls_id]
            self.interval_reward[tls_id] += discount * self.reward_scale * raw_reward
            for key, value in terms.items():
                self.interval_components[tls_id][key] += discount * value
            self.interval_discount[tls_id] *= self.gamma_per_second
            self.interval_duration[tls_id] += 1

    def consume_interval_reward(self, tls_id):
        result = (float(self.interval_reward[tls_id]),
                  int(self.interval_duration[tls_id]),
                  dict(self.interval_components[tls_id]))
        self.reset_interval_reward(tls_id)
        return result

    def reset_interval_reward(self, tls_id):
        self.interval_reward[tls_id] = 0.0
        self.interval_components[tls_id] = defaultdict(float)
        self.interval_discount[tls_id] = 1.0
        self.interval_duration[tls_id] = 0

    def get_all_observations(self):
        return {tls_id: self.get_observation(tls_id) for tls_id in self.net_info}

    def get_observation_matrix(self, tls_ids):
        return np.stack([self.get_observation(tls_id) for tls_id in tls_ids]).astype(np.float32)

    def get_global_context(self, observations):
        matrix = np.asarray(observations, dtype=np.float32)
        return np.concatenate((matrix.mean(axis=0), matrix.max(axis=0))).astype(np.float32)

    def get_observation(self, tls_id):
        info = self.net_info[tls_id]
        action_count = info["phase_num"] // 2
        mp_raw = np.asarray(self._collect_mp_pressure(tls_id), dtype=np.float32)
        cv_raw = np.asarray(self._collect_cvmp_pressure(tls_id), dtype=np.float32)
        mp_norm = np.asarray(self._normalize_pressure(mp_raw), dtype=np.float32)
        cv_norm = np.asarray(self._normalize_pressure(cv_raw), dtype=np.float32)

        phase_features = []
        for action_idx in range(self.max_actions):
            if action_idx >= action_count:
                phase_features.extend([0.0] * 6)
                continue
            lanes = set(info["phase2lane"].get(action_idx * 2, ()))
            lanes.intersection_update(info["lanes_in_no_right"])
            if lanes:
                queue = sum(float(self._lane_value(lane, tc.LAST_STEP_VEHICLE_HALTING_NUMBER))
                            for lane in lanes)
                count = sum(float(self._lane_value(lane, tc.LAST_STEP_VEHICLE_NUMBER))
                            for lane in lanes)
                capacity = sum(self.lane_capacity.get(lane, 1.0) for lane in lanes)
                queue_ratio = queue / max(capacity, 1.0)
                occupancy_ratio = count / max(capacity, 1.0)
                waiting = sum(self.lane_waiting_sum[lane] for lane in lanes)
                waiting_state = (waiting / max(len(lanes), 1) /
                                 max(self.state_waiting_scale, 1e-6))
                max_waiting_state = (max(self.lane_max_waiting[lane]
                                         for lane in lanes) /
                                     max(self.state_waiting_scale, 1e-6))
            else:
                queue_ratio = occupancy_ratio = 0.0
                waiting_state = max_waiting_state = 0.0
            phase_features.extend([float(mp_norm[action_idx]),
                                   float(cv_norm[action_idx]),
                                   float(np.clip(queue_ratio, 0.0, 1.5)),
                                   float(np.clip(occupancy_ratio, 0.0, 1.5)),
                                   float(np.clip(waiting_state, 0.0, 2.0)),
                                   float(np.clip(max_waiting_state, 0.0, 2.0))])

        current_action = int(traci.trafficlight.getPhase(tls_id) // 2)
        current_one_hot = [1.0 if idx == current_action else 0.0
                           for idx in range(self.max_actions)]
        elapsed = max(0.0, traci.simulation.getTime() - self.phase_switch_pointer[tls_id])
        local_queue = self._intersection_queue_ratio(tls_id)
        local_lanes = info["lanes_in_no_right"]
        local_waiting = (sum(self.lane_waiting_sum[lane] for lane in local_lanes) /
                         max(len(local_lanes), 1) /
                         max(self.state_waiting_scale, 1e-6))
        cv_ratio = self._intersection_cv_ratio_cached(tls_id)
        method_conflict = float(int(np.argmax(mp_norm)) != int(np.argmax(cv_norm)))
        mp_margin = self._top_margin(mp_norm)
        cv_margin = self._top_margin(cv_norm)
        tail = current_one_hot + [min(elapsed / 40.0, 1.5),
                                  float(np.clip(local_queue, 0.0, 1.5)),
                                  float(np.clip(local_waiting, 0.0, 2.0)),
                                  cv_ratio,
                                  method_conflict,
                                  mp_margin,
                                  cv_margin]
        return np.asarray(phase_features + tail, dtype=np.float32)

    def _intersection_cv_ratio_cached(self, tls_id):
        vehicle_ids = set()
        for lane in self.net_info[tls_id]["lanes_in"]:
            vehicle_ids.update(self._lane_value(lane, tc.LAST_STEP_VEHICLE_ID_LIST, ()))
        if not vehicle_ids:
            return 0.0
        return sum(veh_id in self.cv_vehicle_ids for veh_id in vehicle_ids) / len(vehicle_ids)

    @staticmethod
    def _top_margin(values):
        if len(values) < 2:
            return 0.0
        ordered = np.sort(values)
        return float(ordered[-1] - ordered[-2])

    def get_action_mask(self, tls_id):
        """Every alpha keeps a fixed, state-independent semantic meaning."""
        return np.ones(len(self.alpha_bins), dtype=bool)

    def get_action_diagnostics(self, tls_id):
        mp = np.asarray(self._normalize_pressure(self._collect_mp_pressure(tls_id)))
        cv = np.asarray(self._normalize_pressure(self._collect_cvmp_pressure(tls_id)))
        phases = [self._select_phase_by_alpha(tls_id, alpha)
                  for alpha in self.alpha_bins]
        return {
            "actor_valid": len(set(phases)) > 1,
            "candidate_phase_count": len(set(phases)),
            "mp_phase": int(np.argmax(mp)),
            "cvmp_phase": int(np.argmax(cv)),
            "phase_by_action": phases,
        }
