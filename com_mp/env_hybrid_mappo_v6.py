from collections import defaultdict

import numpy as np
import traci
import traci.constants as tc

from env_hybrid_mappo import HybridMAPPOEnv


class DetectorSlimHybridMAPPOEnv(HybridMAPPOEnv):
    """V6 detector-feasible state and queue-dynamics reward.

    The fixed-detector channel supplies MP pressure and queue ratios. The CV
    channel supplies CV-MP pressure and the locally observed CV sample rate.
    No vehicle waiting-history feature is exposed to the policy or reward.
    """

    MOVEMENT_TEMPLATE = tuple((direction, turn) for direction in "NSEW"
                              for turn in ("l", "s"))
    PHASE_TEMPLATE = (("N", "S", "s"), ("E", "W", "s"),
                      ("N", "S", "l"), ("E", "W", "l"))

    def __init__(self, *args, queue_reduction_reward_weight=1.0,
                 queue_level_reward_weight=0.10, queue_change_clip=0.0,
                 decision_interval_seconds=10, max_red_seconds=180.0,
                 **kwargs):
        if max_red_seconds <= 0:
            raise ValueError("max_red_seconds must be positive")
        self.max_red_seconds = float(max_red_seconds)
        # Retain the old feature scale, not a maximum-green constraint.
        self.elapsed_observation_scale = 40.0
        self.controller_mode = "ComMP"
        self.topology_specs = {}
        self._clear_control_state()
        super().__init__(
            *args,
            queue_reward_weight=0.0,
            waiting_reward_weight=0.0,
            throughput_reward_weight=0.0,
            switch_penalty=0.0,
            waiting_time_divisor=1.0,
            state_waiting_scale=1.0,
            **kwargs,
        )
        self._decision_interval = max(
            1, int(decision_interval_seconds))
        self.queue_reduction_reward_weight = float(
            queue_reduction_reward_weight)
        self.queue_level_reward_weight = float(queue_level_reward_weight)
        self.queue_change_clip = float(queue_change_clip)
        self.track_waiting_state = False
        self.previous_queue_ratio = defaultdict(float)

    def reset(self, tripinfo_path):
        self._clear_control_state()
        self.previous_queue_ratio = defaultdict(float)
        observations, net_info, decision_flag = super().reset(tripinfo_path)
        self.track_waiting_state = False
        return self.get_all_observations(), net_info, decision_flag

    def begin_control(self):
        if self._decision_interval <= self._yellow:
            raise ValueError("Decision interval must exceed yellow duration")
        super().begin_control()
        now = float(traci.simulation.getTime())
        self._control_active = True
        self.cv_fallback_state = {}
        for tls, info in self.net_info.items():
            self._red_age[tls] = np.zeros(info["phase_num"] // 2)
            self.decision_step[tls] = now
            self.current_phase_action[tls] = traci.trafficlight.getPhase(tls) // 2
            self._freeze_phase(tls)
        self.current_decision_flag = {tls: 1 for tls in self.net_info}
        self.previous_queue_ratio = defaultdict(float, {
            tls_id: self._intersection_queue_ratio(tls_id)
            for tls_id in self.net_info
        })

    def _intersection_queue_ratio(self, tls_id):
        """Halting vehicles divided by capacity of distinct left/through lanes."""
        lanes = set(self.net_info[tls_id]["lanes_in_no_right"])
        queued = sum(float(self._lane_value(
            lane, tc.LAST_STEP_VEHICLE_HALTING_NUMBER)) for lane in lanes)
        capacity = sum(self.lane_capacity.get(lane, 1.0) for lane in lanes)
        return queued / max(capacity, 1.0)

    def _accumulate_reward_step(self):
        for tls_id in self.net_info:
            queue_level = float(self._intersection_queue_ratio(tls_id))
            previous = float(self.previous_queue_ratio[tls_id])
            queue_reduction = previous - queue_level
            if self.queue_change_clip > 0.0:
                queue_reduction = float(np.clip(
                    queue_reduction,
                    -self.queue_change_clip,
                    self.queue_change_clip,
                ))

            raw_reward = (
                self.queue_reduction_reward_weight * queue_reduction -
                self.queue_level_reward_weight * queue_level
            )
            discount = self.interval_discount[tls_id]
            self.interval_reward[tls_id] += (
                discount * self.reward_scale * raw_reward)
            self.interval_components[tls_id]["queue_reduction"] += (
                discount * queue_reduction)
            self.interval_components[tls_id]["queue_level"] += (
                discount * queue_level)
            self.interval_discount[tls_id] *= self.gamma_per_second
            self.interval_duration[tls_id] += 1
            self.previous_queue_ratio[tls_id] = queue_level

    def _clear_control_state(self):
        self._control_active = False
        self._red_age = {}
        self._pending_green = {}
        self._service_cache = {}
        self._service_tick = None
        self._length_cache = {}
        self._template_cache = {}

    def _physical_length(self, lane):
        if lane not in self._length_cache:
            self._length_cache[lane] = max(float(traci.lane.getLength(lane)), 1e-6)
        return self._length_cache[lane]

    def _freeze_phase(self, tls):
        traci.trafficlight.setPhaseDuration(tls, 1e9)

    def _service_data(self):
        tick = float(traci.simulation.getTime())
        if tick != self._service_tick:
            self._service_tick = tick
            self._service_cache = {}
        return self._service_cache

    def _lane_vehicles(self, lane):
        cache = self._service_data()
        key = ("vehicles", lane)
        if key not in cache:
            cache[key] = tuple(self._lane_value(lane, tc.LAST_STEP_VEHICLE_ID_LIST))
        return cache[key]

    def _head_next_edge(self, lane):
        cache = self._service_data()
        key = ("head", lane)
        if key not in cache:
            vehicles = self._lane_vehicles(lane)
            head = max(vehicles, key=traci.vehicle.getLanePosition) if vehicles else None
            cache[key] = (head, self._get_live_vehicle_next_edge(
                head, self._lane_to_edge(lane)) if head else None)
        return cache[key]

    def _receiving_space(self, lane, head):
        """Conservative nominal receiving model, not measured future discharge."""
        cache = self._service_data()
        key = ("space", lane)
        if key not in cache:
            vehicles = self._lane_vehicles(lane)
            free_storage = self._physical_length(lane) - sum(
                traci.vehicle.getLength(v) + traci.vehicle.getMinGap(v)
                for v in vehicles)
            entrance_gap = min((traci.vehicle.getLanePosition(v) -
                                traci.vehicle.getLength(v) for v in vehicles),
                               default=self._physical_length(lane))
            cache[key] = min(free_storage, entrance_gap)
        needed = traci.vehicle.getLength(head) + traci.vehicle.getMinGap(head)
        return cache[key] >= needed

    def _movement_service(self, movement, action):
        head, next_edge = self._head_next_edge(movement["in_lane"])
        if head is None or next_edge != movement["out_edge"]:
            return 0.0
        if not self._receiving_space(movement["out_lane"], head):
            return 0.0
        current = int(traci.trafficlight.getPhase(movement["tls_id"]) // 2)
        horizon = float(self._decision_interval)
        fraction = 1.0 if current == action else max(0.0, horizon - self._yellow) / horizon
        # Existing pressure units use veh/h; both constituents share this rate.
        return float(self.saturation_flow_rate) * fraction

    def _constituent_pressure(self, tls, cv):
        cache = self._service_data()
        key = ("pressure", tls, cv, int(traci.trafficlight.getPhase(tls)))
        if key in cache:
            return cache[key]
        self._ensure_mp_network_info()
        if cv:
            states, counts = self._collect_cv_movement_states()
            pressure_fn = self._movement_pressure
        else:
            states, counts = self._collect_mp_movement_states()
            pressure_fn = self._mp_movement_pressure
        scores = []
        for action, movements in enumerate(self.mp_tls_action_movements[tls]):
            # Multiple SUMO connections to the same edge must not duplicate service.
            by_lane = {}
            for movement in movements:
                if not self._movement_allowed(tls, movement):
                    continue
                value = pressure_fn(movement, states, counts) * self._movement_service(movement, action)
                lane = movement["in_lane"]
                by_lane[lane] = max(by_lane.get(lane, 0.0), value)
            scores.append(sum(by_lane.values()))
        cache[key] = scores
        return scores

    def _collect_mp_pressure(self, tls_id):
        return self._constituent_pressure(tls_id, False)

    def _collect_cvmp_pressure(self, tls_id):
        return self._constituent_pressure(tls_id, True)

    def information_mode(self, tls):
        detector_ids = getattr(self, "detector_tls_ids", None)
        detector = detector_ids is None or tls in detector_ids
        cv = self._intersection_cv_ratio_cached(tls) > 0
        if self.controller_mode == "MP":
            return "forced_mp"
        if self.controller_mode == "CVMP":
            return "forced_cvmp" if cv else "fixed_time"
        if detector and cv:
            return "hybrid"
        if detector:
            return "forced_mp"
        return "forced_cvmp" if cv else "fixed_time"

    def _effective_alpha(self, tls, alpha):
        mode = self.information_mode(tls)
        if mode == "forced_mp":
            return 0.0
        if mode == "forced_cvmp":
            return 1.0
        return float(np.clip(alpha, 0.0, 1.0))

    def _phase_mask(self, tls):
        count = self.net_info[tls]["phase_num"] // 2
        if tls not in self.topology_specs:
            return np.ones(count, dtype=bool)
        _, phase_slots = self._template(tls)
        mask = np.zeros(count, dtype=bool)
        for slot, native in enumerate(phase_slots):
            if native is not None:
                mask[native] = self.topology_specs[tls][1][slot]
        if not mask.any():
            raise ValueError(f"No feasible phase at {tls}")
        return mask

    def _overdue_phase(self, tls):
        ages = self._red_age.get(tls)
        if ages is None:
            return None
        eligible = self._phase_mask(tls) & (ages >= self.max_red_seconds)
        if not eligible.any():
            return None
        return int(np.argmax(np.where(eligible, ages, -np.inf)))

    def _select_phase_by_alpha(self, tls_id, alpha):
        forced = self._overdue_phase(tls_id)
        if forced is not None:
            return forced
        if self.information_mode(tls_id) == "fixed_time":
            return int(traci.trafficlight.getPhase(tls_id) // 2)
        alpha = self._effective_alpha(tls_id, alpha)
        pressure = np.asarray(self._collect_hybrid_pressure(tls_id, alpha), dtype=float)
        return int(np.argmax(np.where(self._phase_mask(tls_id), pressure, -np.inf)))

    def get_action_diagnostics(self, tls_id):
        phases = [self._select_phase_by_alpha(tls_id, a) for a in self.alpha_bins]
        return {"actor_valid": len(set(phases)) > 1,
                "candidate_phase_count": len(set(phases)),
                "mp_phase": self._select_phase_by_alpha(tls_id, 0.0),
                "cvmp_phase": self._select_phase_by_alpha(tls_id, 1.0),
                "phase_by_action": phases}

    def _switch_to(self, tls, action, now):
        current = int(traci.trafficlight.getPhase(tls) // 2)
        self.current_phase_action[tls] = int(action)
        if action == current:
            self._freeze_phase(tls)
            return
        traci.trafficlight.setPhase(tls, current * 2 + 1)
        self._freeze_phase(tls)
        self._pending_green[tls] = (now + self._yellow, int(action))
        self.pending_switch_events[tls] += 1

    def _fixed_control(self, tls, now):
        valid = np.flatnonzero(self._phase_mask(tls)).tolist()
        green = (100.0 - len(valid) * self._yellow) / len(valid)
        if green <= 0:
            raise ValueError("Too many phases for a 100-second fallback cycle")
        state = self.cv_fallback_state.get(tls)
        if state is None:
            current = int(traci.trafficlight.getPhase(tls) // 2)
            if current not in valid:
                current = valid[0]
                self._switch_to(tls, current, now)
            state = {"cycle_start_time": now, "cycle_end_time": now + 100.0,
                     "reason": "no_information", "next_switch": now + green}
            self.cv_fallback_state[tls] = state
        if now >= state["next_switch"] and tls not in self._pending_green:
            current = int(traci.trafficlight.getPhase(tls) // 2)
            target = valid[(valid.index(current) + 1) % len(valid)]
            self._switch_to(tls, target, now)
            state["next_switch"] += green + self._yellow
        if now >= state["cycle_end_time"]:
            state["cycle_start_time"] = state["cycle_end_time"]
            state["cycle_end_time"] += 100.0

    def step(self, alpha_action):
        if not self._control_active:
            self.begin_control()
        now = float(traci.simulation.getTime())
        self.control_input = {}
        for tls in self.net_info:
            if now >= self.decision_step[tls] and tls not in self._pending_green:
                mode = self.information_mode(tls)
                forced = self._overdue_phase(tls)
                alpha = self._effective_alpha(tls, alpha_action.get(tls, self.last_alpha[tls]))
                self.last_alpha[tls] = alpha
                if mode != "fixed_time" or forced is not None:
                    self.cv_fallback_state.pop(tls, None)
                    self._switch_to(tls, self._select_phase_by_alpha(tls, alpha), now)
                else:
                    self._fixed_control(tls, now)
                self.decision_step[tls] = now + self._decision_interval
            if tls in self.cv_fallback_state:
                self._fixed_control(tls, now)
            self.control_input[tls] = self.current_phase_action[tls]
        before = {tls: int(traci.trafficlight.getPhase(tls)) for tls in self.net_info}
        self._simulate(1)
        after = float(traci.simulation.getTime())
        for tls, phase in before.items():
            self._red_age[tls] += after - now
            if phase % 2 == 0:
                self._red_age[tls][phase // 2] = 0.0
            pending = self._pending_green.get(tls)
            if pending is not None and after >= pending[0]:
                traci.trafficlight.setPhase(tls, pending[1] * 2)
                self._freeze_phase(tls)
                self.phase_switch_pointer[tls] = after
                self._red_age[tls][pending[1]] = 0.0
                del self._pending_green[tls]
        self.next_decision_flag = {tls: int(after >= self.decision_step[tls] and
                                           tls not in self._pending_green)
                                   for tls in self.net_info}
        self.current_decision_flag = self.next_decision_flag
        dones = {tls: [self._step >= self._max_steps - 15] for tls in self.net_info}
        return {}, dones, self.next_decision_flag, {}

    def _template(self, tls):
        if tls not in self._template_cache:
            self._ensure_mp_network_info()
            lanes = [[] for _ in self.MOVEMENT_TEMPLATE]
            for lane in self.net_info[tls]["lanes_in_no_right"]:
                pair = (self._lane_dir(lane), self._lane_turn(lane))
                if pair in self.MOVEMENT_TEMPLATE:
                    lanes[self.MOVEMENT_TEMPLATE.index(pair)].append(lane)
            phases = [None] * 4
            for native, movements in enumerate(self.mp_tls_action_movements[tls]):
                signatures = {(self._lane_dir(m["in_lane"]), self._lane_turn(m["in_lane"]))
                              for m in movements}
                if not signatures:
                    continue
                matches = [i for i, (a, b, turn) in enumerate(self.PHASE_TEMPLATE)
                           if signatures <= {(a, turn), (b, turn)}]
                if len(matches) != 1 or phases[matches[0]] is not None:
                    raise ValueError(f"{tls}: extend the common topology template for this phase structure")
                phases[matches[0]] = native
            self._template_cache[tls] = (lanes, phases)
        return self._template_cache[tls]

    def set_topology_mask(self, tls, movement_mask=None, phase_mask=None):
        """Opt-in canonical masks; None derives missing entries from the network."""
        lanes, phases = self._template(tls)
        present_m = np.asarray([bool(v) for v in lanes])
        present_p = np.asarray([v is not None for v in phases])
        def validate(value, present):
            if value is None:
                return present
            value = np.asarray(value)
            if value.shape != present.shape or not np.isin(value, [0, 1]).all():
                raise ValueError("Topology masks must be binary and match the common template")
            return value.astype(bool) & present
        m = validate(movement_mask, present_m)
        p = validate(phase_mask, present_p)
        for slot, (a, b, turn) in enumerate(self.PHASE_TEMPLATE):
            p[slot] &= any(m[i] for i, pair in enumerate(self.MOVEMENT_TEMPLATE)
                           if pair in ((a, turn), (b, turn)))
        if not p.any():
            raise ValueError("At least one feasible phase is required")
        self.topology_specs[tls] = (m, p)
        self._service_cache.clear()

    def _movement_allowed(self, tls, movement):
        if tls not in self.topology_specs:
            return True
        lanes, _ = self._template(tls)
        return any(movement["in_lane"] in group and self.topology_specs[tls][0][i]
                   for i, group in enumerate(lanes))

    def get_observation(self, tls):
        """18 slots: four phase Q/CV/queue triples, phase one-hot, time and CV."""
        lanes, phases = self._template(tls)
        if tls not in self.topology_specs and (not all(lanes) or None in phases):
            self.set_topology_mask(tls)
        phase_mask = self.topology_specs.get(
            tls, (np.ones(8, bool), np.ones(4, bool)))[1]
        mp = np.asarray(self._collect_mp_pressure(tls), dtype=float)
        cv = np.asarray(self._collect_cvmp_pressure(tls), dtype=float)
        pressures = []
        for values in (mp, cv):
            ordered = np.asarray([values[p] if p is not None else 0.0
                                  for p in phases], dtype=float)
            valid = ordered[phase_mask]
            lo, hi = (valid.min(), valid.max()) if valid.size else (0.0, 0.0)
            ordered = (ordered - lo) / (hi - lo) if hi > lo else np.zeros(4)
            ordered[~phase_mask] = 0.0
            pressures.append(ordered)

        features = []
        info = self.net_info[tls]
        for slot, native in enumerate(phases):
            if native is None or not phase_mask[slot]:
                features.extend([0.0, 0.0, 0.0])
                continue
            served_lanes = set(info["phase2lane"].get(native * 2, ()))
            served_lanes.intersection_update(info["lanes_in_no_right"])
            if tls in self.topology_specs:
                movement_mask = self.topology_specs[tls][0]
                allowed = {lane for i, group in enumerate(lanes)
                           if movement_mask[i] for lane in group}
                served_lanes.intersection_update(allowed)
            capacity = sum(self.lane_capacity.get(lane, 1.0) for lane in served_lanes)
            queued = sum(self._lane_value(lane, tc.LAST_STEP_VEHICLE_HALTING_NUMBER)
                         for lane in served_lanes)
            features.extend([pressures[0][slot], pressures[1][slot],
                             float(np.clip(queued / max(capacity, 1.0), 0.0, 1.5))])
        current = int(traci.trafficlight.getPhase(tls) // 2)
        onehot = [float(native == current and phase_mask[i])
                  for i, native in enumerate(phases)]
        elapsed = max(0, traci.simulation.getTime() - self.phase_switch_pointer[tls])
        return np.asarray(features + onehot +
                          [min(elapsed / self.elapsed_observation_scale, 1.5),
                           self._intersection_cv_ratio_cached(tls)], dtype=np.float32)
