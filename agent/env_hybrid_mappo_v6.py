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

    def __init__(self, *args, queue_reduction_reward_weight=1.0,
                 queue_level_reward_weight=0.10, queue_change_clip=0.25,
                 decision_interval_seconds=10,
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
        self.queue_change_clip = float(queue_change_clip)
        self.track_waiting_state = False
        self.previous_queue_ratio = defaultdict(float)

    def reset(self, tripinfo_path):
        self.previous_queue_ratio = defaultdict(float)
        observations, net_info, decision_flag = super().reset(tripinfo_path)
        self.track_waiting_state = False
        return self.get_all_observations(), net_info, decision_flag

    def begin_control(self):
        super().begin_control()
        self.previous_queue_ratio = defaultdict(float, {
            tls_id: self._intersection_queue_ratio(tls_id)
            for tls_id in self.net_info
        })

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

    def get_observation(self, tls_id):
        """Return 4M+2 deployable features; M=4 gives 18 dimensions."""
        info = self.net_info[tls_id]
        action_count = info["phase_num"] // 2
        mp_norm = np.asarray(self._normalize_pressure(
            self._collect_mp_pressure(tls_id)), dtype=np.float32)
        cv_norm = np.asarray(self._normalize_pressure(
            self._collect_cvmp_pressure(tls_id)), dtype=np.float32)

        phase_features = []
        for action_idx in range(self.max_actions):
            if action_idx >= action_count:
                phase_features.extend([0.0, 0.0, 0.0])
                continue

            lanes = set(info["phase2lane"].get(action_idx * 2, ()))
            lanes.intersection_update(info["lanes_in_no_right"])
            if lanes:
                queued = sum(float(self._lane_value(
                    lane, tc.LAST_STEP_VEHICLE_HALTING_NUMBER))
                    for lane in lanes)
                capacity = sum(self.lane_capacity.get(lane, 1.0)
                               for lane in lanes)
                queue_ratio = queued / max(capacity, 1.0)
            else:
                queue_ratio = 0.0

            phase_features.extend([
                float(mp_norm[action_idx]),
                float(cv_norm[action_idx]),
                float(np.clip(queue_ratio, 0.0, 1.5)),
            ])

        current_action = int(traci.trafficlight.getPhase(tls_id) // 2)
        current_one_hot = [
            1.0 if idx == current_action else 0.0
            for idx in range(self.max_actions)
        ]
        elapsed = max(
            0.0,
            traci.simulation.getTime() - self.phase_switch_pointer[tls_id],
        )
        elapsed_ratio = min(elapsed / max(self._green_max, 1), 1.5)
        local_cv_ratio = self._intersection_cv_ratio_cached(tls_id)
        tail = current_one_hot + [elapsed_ratio, local_cv_ratio]
        return np.asarray(phase_features + tail, dtype=np.float32)
