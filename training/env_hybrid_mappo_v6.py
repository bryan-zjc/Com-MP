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

    MOVEMENT_SLOTS = (
        ("E", "s"), ("E", "l"),
        ("S", "s"), ("S", "l"),
        ("W", "s"), ("W", "l"),
        ("N", "s"), ("N", "l"),
    )
    STANDARD_PHASE_COUNT = 4

    def __init__(self, *args, queue_reduction_reward_weight=1.0,
                 queue_level_reward_weight=0.10,
                 decision_interval_seconds=10,
                 route_defined_cv=False,
                 stop_exactly_at_max_steps=False,
                 **kwargs):
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
        self.track_waiting_state = False
        self.previous_movement_queue_ratios = {}
        self._topology_mask_cache = {}
        self.route_defined_cv = bool(route_defined_cv)
        self.stop_exactly_at_max_steps = bool(stop_exactly_at_max_steps)

    def reset(self, tripinfo_path):
        self.previous_movement_queue_ratios = {}
        self._topology_mask_cache = {}
        observations, net_info, decision_flag = super().reset(tripinfo_path)
        self.track_waiting_state = False
        return self.get_all_observations(), net_info, decision_flag

    def begin_control(self):
        super().begin_control()
        self.previous_movement_queue_ratios = {
            tls_id: self._movement_queue_ratios(tls_id)
            for tls_id in self.net_info
        }

    def _is_cv_vehicle(self, veh_id):
        """Honor stage-specific CV/NV identities encoded in dynamic routes."""
        if not self.route_defined_cv:
            return super()._is_cv_vehicle(veh_id)
        if veh_id in self.cv_vehicle_ids:
            return True
        if veh_id in self.nv_vehicle_ids:
            return False
        if veh_id.endswith("_CV"):
            self.cv_vehicle_ids.add(veh_id)
            return True
        if veh_id.endswith("_NV"):
            self.nv_vehicle_ids.add(veh_id)
            return False
        is_cv = traci.vehicle.getTypeID(veh_id) == "CV"
        (self.cv_vehicle_ids if is_cv else self.nv_vehicle_ids).add(veh_id)
        return is_cv

    def step(self, alpha_action):
        result = super().step(alpha_action)
        if not self.stop_exactly_at_max_steps:
            return result
        pressures, dones, decision_flags, info = result
        dones = {
            tls_id: [self._step >= self._max_steps]
            for tls_id in self.net_info
        }
        return pressures, dones, decision_flags, info

    def _movement_lane_groups(self, tls_id):
        """Map controlled lanes to the common eight-movement template."""
        grouped_lanes = {slot: set() for slot in self.MOVEMENT_SLOTS}
        for lane_id in self.net_info[tls_id]["lanes_in_no_right"]:
            slot = (self._lane_dir(lane_id, incoming=True),
                    self._lane_turn(lane_id))
            if slot in grouped_lanes:
                grouped_lanes[slot].add(lane_id)
        return grouped_lanes

    def _movement_queue_ratios(self, tls_id):
        """Eight fixed-order movement queue-occupancy ratios."""
        grouped_lanes = self._movement_lane_groups(tls_id)

        ratios = np.zeros(len(self.MOVEMENT_SLOTS), dtype=np.float32)
        for index, slot in enumerate(self.MOVEMENT_SLOTS):
            lanes = grouped_lanes[slot]
            if not lanes:
                continue
            queued = sum(float(self._lane_value(
                lane_id, tc.LAST_STEP_VEHICLE_HALTING_NUMBER))
                for lane_id in lanes)
            capacity = sum(float(self.lane_capacity.get(lane_id, 1.0))
                           for lane_id in lanes)
            ratios[index] = queued / max(capacity, 1.0)
        return ratios

    def _build_topology_mask(self, tls_id):
        """Build the binary mask for the fixed observation template."""
        action_count = self.net_info[tls_id]["phase_num"] // 2
        phase_mask = np.zeros(self.max_actions, dtype=np.float32)
        phase_mask[:min(action_count, self.max_actions)] = 1.0

        grouped_lanes = self._movement_lane_groups(tls_id)
        movement_mask = np.asarray(
            [bool(grouped_lanes[slot]) for slot in self.MOVEMENT_SLOTS],
            dtype=np.float32,
        )
        observation_mask = np.concatenate((
            phase_mask,       # Q-MP phase-pressure vector
            phase_mask,       # CV-MP phase-pressure vector
            movement_mask,    # movement queue-occupancy ratios
            phase_mask,       # current-phase one-hot vector
            np.ones(2, dtype=np.float32),  # elapsed green and local CV ratio
        )).astype(np.float32)
        return {
            "required": bool(
                action_count != self.STANDARD_PHASE_COUNT or
                np.any(movement_mask == 0.0)
            ),
            "observation": observation_mask,
            "phase": phase_mask,
            "movement": movement_mask,
        }

    def _topology_mask_info(self, tls_id):
        if tls_id not in self._topology_mask_cache:
            self._topology_mask_cache[tls_id] = (
                self._build_topology_mask(tls_id))
        return self._topology_mask_cache[tls_id]

    def requires_topology_mask(self, tls_id):
        """Return False for a complete 8-movement, 4-phase intersection."""
        return self._topology_mask_info(tls_id)["required"]

    def get_topology_mask(self, tls_id):
        """Expose the fixed-length binary observation mask."""
        return self._topology_mask_info(tls_id)["observation"].copy()

    def apply_topology_mask(self, tls_id, observation):
        """Exclude padded topology entries while preserving input length."""
        values = np.asarray(observation, dtype=np.float32)
        mask = self.get_topology_mask(tls_id)
        if values.shape != mask.shape:
            raise ValueError(
                f"Observation/mask shape mismatch at {tls_id}: "
                f"{values.shape} != {mask.shape}")
        return values * mask

    def _accumulate_reward_step(self):
        for tls_id in self.net_info:
            current_ratios = self._movement_queue_ratios(tls_id)
            previous_ratios = self.previous_movement_queue_ratios[tls_id]
            queue_reduction = float(np.sum(
                previous_ratios - current_ratios))
            queue_level = float(np.sum(current_ratios))

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
            self.previous_movement_queue_ratios[tls_id] = current_ratios

    def get_observation(self, tls_id):
        """Return [Q pressures, CV pressures, movement queues, phase, time, CV]."""
        info = self.net_info[tls_id]
        action_count = info["phase_num"] // 2
        mp_norm = np.asarray(self._normalize_pressure(
            self._collect_mp_pressure(tls_id)), dtype=np.float32)
        cv_norm = np.asarray(self._normalize_pressure(
            self._collect_cvmp_pressure(tls_id)), dtype=np.float32)

        mp_features = np.zeros(self.max_actions, dtype=np.float32)
        cv_features = np.zeros(self.max_actions, dtype=np.float32)
        mp_count = min(self.max_actions, action_count, len(mp_norm))
        cv_count = min(self.max_actions, action_count, len(cv_norm))
        mp_features[:mp_count] = mp_norm[:mp_count]
        cv_features[:cv_count] = cv_norm[:cv_count]
        movement_features = self._movement_queue_ratios(tls_id)

        current_action = int(traci.trafficlight.getPhase(tls_id) // 2)
        current_one_hot = [
            1.0 if idx == current_action else 0.0
            for idx in range(self.max_actions)
        ]
        elapsed = max(
            0.0,
            traci.simulation.getTime() - self.phase_switch_pointer[tls_id],
        )
        elapsed_ratio = min(elapsed / max(self._max_red, 1), 1.5)
        local_cv_ratio = self._intersection_cv_ratio_cached(tls_id)
        observation = np.asarray(
            mp_features.tolist() +
            cv_features.tolist() +
            movement_features.tolist() +
            current_one_hot +
            [elapsed_ratio, local_cv_ratio],
            dtype=np.float32,
        )
        if self.requires_topology_mask(tls_id):
            return self.apply_topology_mask(tls_id, observation)
        return observation
