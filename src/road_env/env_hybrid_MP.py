# -*- coding: utf-8 -*-
"""
Hybrid MP environment.

Pressure used for execution:
    hybrid_pressure = alpha * minmax(CV_MP) + (1 - alpha) * minmax(MP)

The underlying MP and CV-MP pressure definitions are reused from env_MP.py and
env_CVMP.py, so existing MP/CVMP entry points are not changed.
"""

from collections import defaultdict

import numpy as np
import traci

from src.road_env.env_MP import Intersec_Env as MP_Intersec_Env
from src.road_env.env_CVMP import Intersec_Env as CVMP_Intersec_Env


class HybridMPEnv(CVMP_Intersec_Env):
    def __init__(self,
                 rou_file,
                 net_file,
                 num_agent,
                 sumo_cmd,
                 max_steps,
                 seed,
                 cv_penetration_rate=1.0,
                 decision_interval_seconds=10):
        super(HybridMPEnv, self).__init__(rou_file,
                                          net_file,
                                          num_agent,
                                          sumo_cmd,
                                          max_steps,
                                          seed,
                                          cv_penetration_rate)
        self._decision_interval = max(
            1, int(decision_interval_seconds))
        # Pure CV-MP may enable the no-CV fixed-time fallback. Hybrid MP keeps
        # its learned alpha behavior unless an evaluator explicitly enables it
        # for the alpha=1 CV-MP baseline.
        self.cv_absence_fixed_time_fallback = False
        self.alpha_default = 0.5
        self.pressure_scale = 1800 * 50
        self.pressure_norm_eps = 1e-6
        self.count_scale = 20.0
        self.wait_scale = 300.0
        self.waiting_speed_threshold = 0.1
        self.hybrid_state_ready = False
        self.last_alpha = {}
        self.current_phase_action = {}
        self.current_decision_flag = {}
        self._reset_lane_waiting_records()

    def _reset_lane_waiting_records(self):
        self.vehicle_wait_lane = {}
        self.vehicle_lane_waiting_time = {}

    def reset(self, tripinfo_path):
        self.last_alpha = {}
        self.current_phase_action = {}
        self.current_decision_flag = {}
        self.hybrid_state_ready = False
        self._reset_lane_waiting_records()

        _, net_info = super().reset(tripinfo_path)

        for tls_id in self.net_info.keys():
            self.last_alpha[tls_id] = self.alpha_default
            self.current_phase_action[tls_id] = 0
            self.current_decision_flag[tls_id] = 0

        states = self.get_all_states()
        return states, net_info, self.current_decision_flag

    def step(self, alpha_action):
        self.control_input = {}
        current_time = traci.simulation.getTime()
        decision_flag = {}

        for tls_id in self.net_info.keys():
            decision_flag[tls_id] = int(self.decision_step[tls_id] == current_time)

            if self.cv_absence_fixed_time_fallback:
                proposed_action = self.current_phase_action[tls_id]
                if decision_flag[tls_id] == 1:
                    alpha = float(alpha_action.get(
                        tls_id, self.last_alpha[tls_id]))
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    self.last_alpha[tls_id] = alpha
                    proposed_action = self._select_phase_by_alpha(
                        tls_id, alpha)
                if self.last_alpha[tls_id] >= 1.0 - 1e-9:
                    fallback_action = self._resolve_cv_absence_fallback(
                        tls_id, proposed_action,
                        decision_flag[tls_id] == 1)
                    if fallback_action is not None:
                        self.current_phase_action[tls_id] = int(
                            fallback_action)
                        self.control_input[tls_id] = fallback_action
                        continue
                elif tls_id in self.cv_fallback_state:
                    self._exit_cv_absence_fallback(tls_id)

            if self.g_max_flag[tls_id] == 1:
                current_phase = traci.trafficlight.getPhase(tls_id)
                self.control_input[tls_id] = ((current_phase + 1) %
                                              self.net_info[tls_id]['phase_num']) / 2
                self.current_phase_action[tls_id] = int(self.control_input[tls_id])
                continue

            if decision_flag[tls_id] == 1:
                alpha = float(alpha_action.get(tls_id, self.last_alpha[tls_id]))
                alpha = float(np.clip(alpha, 0.0, 1.0))
                self.last_alpha[tls_id] = alpha
                self.current_phase_action[tls_id] = self._select_phase_by_alpha(tls_id, alpha)

            self.control_input[tls_id] = self.current_phase_action[tls_id]

        self._update_env()
        self._finalize_cv_absence_fallback()

        dones = {}
        for tls_id in self.net_info.keys():
            dones[tls_id] = [self._step >= self._max_steps - 15]

        self.current_decision_flag = self.next_decision_flag
        return {}, dones, self.next_decision_flag, {}

    def _simulate(self, steps_todo=1):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            previous_time = traci.simulation.getTime()
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            step_duration = max(current_time - previous_time, 0.0)
            if step_duration <= 0.0:
                step_duration = 1.0

            self._step += 1
            steps_todo -= 1
            self._update_cv_vehicle_states()
            self._update_lane_waiting_times(step_duration)

        return self._step

    def _update_lane_waiting_times(self, step_duration):
        tracked_vehicle_ids = set()

        for lane_id in self._get_wait_tracking_lanes():
            for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                tracked_vehicle_ids.add(veh_id)
                if self.vehicle_wait_lane.get(veh_id) != lane_id:
                    self.vehicle_wait_lane[veh_id] = lane_id
                    self.vehicle_lane_waiting_time[veh_id] = 0.0

                if traci.vehicle.getSpeed(veh_id) < self.waiting_speed_threshold:
                    self.vehicle_lane_waiting_time[veh_id] += step_duration

        for veh_id in list(self.vehicle_wait_lane.keys()):
            if veh_id not in tracked_vehicle_ids:
                self.vehicle_wait_lane.pop(veh_id, None)
                self.vehicle_lane_waiting_time.pop(veh_id, None)

    def _get_wait_tracking_lanes(self):
        if not self.net_info:
            return []

        lanes = set()
        for tls_info in self.net_info.values():
            lanes.update(tls_info.get('lanes_in', []))
        return lanes

    def _get_current_lane_waiting_time(self, veh_id, lane_id):
        if self.vehicle_wait_lane.get(veh_id) != lane_id:
            return 0.0
        return self.vehicle_lane_waiting_time.get(veh_id, 0.0)

    def _collect_pressure(self, tls_id):
        return self._collect_hybrid_pressure(tls_id, self.last_alpha.get(tls_id, self.alpha_default))

    def _collect_mp_pressure(self, tls_id):
        return MP_Intersec_Env._collect_pressure(self, tls_id)

    def _collect_cvmp_pressure(self, tls_id):
        return CVMP_Intersec_Env._collect_pressure(self, tls_id)

    def _collect_hybrid_pressure(self, tls_id, alpha):
        mp_pressure = self._collect_mp_pressure(tls_id)
        cvmp_pressure = self._collect_cvmp_pressure(tls_id)
        mp_pressure = self._normalize_pressure(mp_pressure)
        cvmp_pressure = self._normalize_pressure(cvmp_pressure)
        return [
            alpha * cv_p + (1.0 - alpha) * mp_p
            for cv_p, mp_p in zip(cvmp_pressure, mp_pressure)
        ]

    def _normalize_pressure(self, pressure):
        pressure_array = np.asarray(pressure, dtype=np.float64)
        pressure_min = float(np.min(pressure_array))
        pressure_max = float(np.max(pressure_array))
        scale = pressure_max - pressure_min
        if scale < self.pressure_norm_eps:
            return np.zeros_like(pressure_array).tolist()
        return ((pressure_array - pressure_min) / scale).tolist()

    def _select_phase_by_alpha(self, tls_id, alpha):
        pressure = self._collect_hybrid_pressure(tls_id, alpha)
        return int(pressure.index(max(pressure)))

    def _ensure_hybrid_state_info(self):
        if self.hybrid_state_ready:
            return

        self.tls_lane_link_indices = defaultdict(lambda: defaultdict(list))
        for tls_id in self.net_info.keys():
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            for link_index, link_group in enumerate(controlled_links):
                for link in link_group:
                    in_lane = link[0]
                    if in_lane and not in_lane.startswith(':'):
                        self.tls_lane_link_indices[tls_id][in_lane].append(link_index)

        self.hybrid_state_ready = True

    def get_all_states(self):
        return {
            tls_id: self.get_state(tls_id)
            for tls_id in self.net_info.keys()
        }

    def get_state_matrix(self, tls_ids):
        return np.vstack([self.get_state(tls_id) for tls_id in tls_ids]).astype(np.float32)

    def get_state(self, tls_id):
        self._ensure_hybrid_state_info()

        state = []
        current_phase = traci.trafficlight.getPhase(tls_id)
        signal_state = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases[current_phase].state

        for lane_id in self.net_info[tls_id]['lanes_in_no_right']:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            waiting_vehicle_num = traci.lane.getLastStepHaltingNumber(lane_id)
            traffic_volume = traci.lane.getLastStepVehicleNumber(lane_id)

            total_waiting_time = 0.0
            first_vehicle_waiting_time = 0.0
            first_vehicle_pos = -1.0
            for veh_id in veh_ids:
                veh_wait = self._get_current_lane_waiting_time(veh_id, lane_id)
                total_waiting_time += veh_wait
                lane_pos = traci.vehicle.getLanePosition(veh_id)
                if lane_pos > first_vehicle_pos:
                    first_vehicle_pos = lane_pos
                    first_vehicle_waiting_time = veh_wait

            phase_of_lane = self._lane_green_flag(tls_id, lane_id, signal_state)
            state.extend([
                waiting_vehicle_num / self.count_scale,
                traffic_volume / self.count_scale,
                total_waiting_time / self.wait_scale,
                first_vehicle_waiting_time / self.wait_scale,
                phase_of_lane,
            ])

        state.append(self._get_intersection_cv_ratio(tls_id))

        mp_pressure = self._collect_mp_pressure(tls_id)
        cvmp_pressure = self._collect_cvmp_pressure(tls_id)
        state.extend(self._normalize_pressure(mp_pressure))
        state.extend(self._normalize_pressure(cvmp_pressure))
        return np.asarray(state, dtype=np.float32)

    def _get_intersection_cv_ratio(self, tls_id):
        total_vehicle_num = 0
        cv_vehicle_num = 0

        for lane_id in self.net_info[tls_id]['lanes_in']:
            for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                total_vehicle_num += 1
                if self._is_cv_vehicle(veh_id):
                    cv_vehicle_num += 1

        return cv_vehicle_num / total_vehicle_num if total_vehicle_num > 0 else 0.0

    def _lane_green_flag(self, tls_id, lane_id, signal_state):
        for link_index in self.tls_lane_link_indices[tls_id].get(lane_id, []):
            if link_index < len(signal_state) and signal_state[link_index] in ('G', 'g'):
                return 1.0
        return 0.0

    def get_average_delay(self, tls_id):
        total_delay = 0.0
        veh_count = 0

        for lane_id in self.net_info[tls_id]['lanes_in_no_right']:
            for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                total_delay += self._get_current_lane_waiting_time(veh_id, lane_id)
                veh_count += 1

        return total_delay / veh_count if veh_count > 0 else 0.0

    def get_delay_dict(self):
        return {
            tls_id: self.get_average_delay(tls_id)
            for tls_id in self.net_info.keys()
        }
