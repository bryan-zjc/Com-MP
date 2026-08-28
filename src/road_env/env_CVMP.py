# -*- coding: utf-8 -*-
"""
Created for CV-MP control.

The signal timing process follows env_MP.py. This file only replaces the
pressure calculation with CV-based normalized link travel time.
"""

import random
from collections import defaultdict
import traci

from src.road_env.env_MP import Intersec_Env as MP_Intersec_Env


class Intersec_Env(MP_Intersec_Env):
    def __init__(self, rou_file, net_file, num_agent, sumo_cmd, max_steps, seed,
                 cv_penetration_rate=1.0,
                 cv_absence_fixed_time_fallback=True,
                 fallback_cycle_seconds=100):
        super(Intersec_Env, self).__init__(rou_file,
                                           net_file,
                                           num_agent,
                                           sumo_cmd,
                                           max_steps,
                                           seed)
        self.cv_penetration_rate = max(0.0, min(1.0, float(cv_penetration_rate)))
        self.cv_absence_fixed_time_fallback = bool(
            cv_absence_fixed_time_fallback)
        self.fallback_cycle_seconds = max(
            1, int(fallback_cycle_seconds))
        self.saturation_flow_rate = 1800
        self._decision_interval = 10
        self.cvmp_network_ready = False
        self._reset_cv_records()

    def _reset_cv_records(self):
        self.cv_random = random.Random(self.seed)
        self.cv_vehicle_ids = set()
        self.nv_vehicle_ids = set()
        self.vehicle_lane = {}
        self.vehicle_edge = {}
        self.vehicle_edge_enter_time = {}
        self.edge_free_flow_time = {}
        self.active_cv_ids = set()
        self.vehicle_route = {}
        self.vehicle_route_index = {}
        self.vehicle_movement_key = {}
        self.vehicle_tau = {}
        self.movement_states = defaultdict(float)
        self.movement_counts = defaultdict(int)
        self.last_pressure = {}
        self.pressure_cache_time = None
        self.pressure_cache = {}
        self.downstream_state_cache_time = None
        self.downstream_state_cache = {}
        self.cv_fallback_state = {}
        self.cvmp_phase_serviceability = {}
        self.cvmp_raw_pressure = {}

    def _lane_to_edge(self, lane_id):
        return lane_id.rsplit('_', 1)[0]

    def reset(self, CVMPtripinfo_path):
        self.cvmp_network_ready = False
        self._reset_cv_records()
        total_pressure, net_info = super().reset(CVMPtripinfo_path)
        self.last_pressure = total_pressure
        return total_pressure, net_info

    def step(self, action):
        self.control_input = {}
        current_time = traci.simulation.getTime()
        for tls_id in self.net_info.keys():
            decision_now = self.decision_step[tls_id] == current_time
            fallback_action = self._resolve_cv_absence_fallback(
                tls_id, action[tls_id], decision_now)
            if fallback_action is not None:
                self.control_input[tls_id] = fallback_action
                continue

            if self.g_max_flag[tls_id] == 1:
                current_phase = traci.trafficlight.getPhase(tls_id)
                self.control_input[tls_id] = ((current_phase + 1) % self.net_info[tls_id]['phase_num']) / 2
            else:
                self.control_input[tls_id] = action[tls_id]

        self._update_env()
        self._finalize_cv_absence_fallback()

        total_pressure = {}
        dones = {}
        for tls_id in self.net_info.keys():
            if self.next_decision_flag.get(tls_id, 0) == 1:
                total_pressure[tls_id] = self._collect_pressure(tls_id)
                self.last_pressure[tls_id] = total_pressure[tls_id]
            else:
                total_pressure[tls_id] = self.last_pressure.get(tls_id, self._empty_pressure(tls_id))
            dones[tls_id] = [self._step >= self._max_steps - 15]

        return total_pressure, dones, self.next_decision_flag

    def _intersection_cv_count(self, tls_id):
        vehicle_ids = set()
        for lane_id in set(self.net_info[tls_id]['lanes_in']):
            vehicle_ids.update(
                traci.lane.getLastStepVehicleIDs(lane_id))
        return sum(
            1 for veh_id in vehicle_ids
            if self._is_cv_vehicle(veh_id))

    def _head_vehicle(self, lane_id):
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        if not vehicle_ids:
            return None

        head_vehicle = None
        head_position = float('-inf')
        for veh_id in vehicle_ids:
            try:
                lane_position = traci.vehicle.getLanePosition(veh_id)
            except traci.TraCIException:
                continue
            if lane_position > head_position:
                head_vehicle = veh_id
                head_position = lane_position
        return head_vehicle

    def _get_live_vehicle_next_edge(self, veh_id, current_edge):
        try:
            route = tuple(traci.vehicle.getRoute(veh_id))
            route_index = int(traci.vehicle.getRouteIndex(veh_id))
        except traci.TraCIException:
            return None

        if (0 <= route_index < len(route) and
                route[route_index] == current_edge):
            next_index = route_index + 1
            return route[next_index] if next_index < len(route) else None

        start_index = max(route_index, 0)
        for edge_index in range(start_index, len(route)):
            if route[edge_index] == current_edge:
                next_index = edge_index + 1
                return route[next_index] if next_index < len(route) else None
        return None

    def _phase_heads_can_pass(self, phase_movements,
                              movement_pressure_by_key):
        served_out_edges = defaultdict(set)
        for movement in phase_movements:
            if movement_pressure_by_key.get(movement['key'], 0.0) <= 1e-9:
                continue
            served_out_edges[movement['in_lane']].add(
                movement['out_edge'])

        if not served_out_edges:
            return False

        # A phase is serviceable only when every lane contributing positive
        # pressure has a head vehicle whose next edge is green in this phase.
        for lane_id, green_out_edges in served_out_edges.items():
            head_vehicle = self._head_vehicle(lane_id)
            if head_vehicle is None:
                return False
            next_edge = self._get_live_vehicle_next_edge(
                head_vehicle, self._lane_to_edge(lane_id))
            if next_edge not in green_out_edges:
                return False
        return True

    def _needs_fixed_cycle(self, tls_id):
        pressure = self._collect_pressure(tls_id)
        return not pressure or max(pressure) <= 1e-9

    def _fallback_green_durations(self, tls_id):
        action_count = self.net_info[tls_id]['phase_num'] // 2
        yellow_total = self._yellow * action_count
        green_budget = self.fallback_cycle_seconds - yellow_total
        if green_budget < self._green_min * action_count:
            raise ValueError(
                f"fallback_cycle_seconds={self.fallback_cycle_seconds} "
                f"is too short for {action_count} phases, "
                f"{self._yellow}s yellow, and "
                f"{self._green_min}s minimum green")
        base_green, remainder = divmod(green_budget, action_count)
        return [
            base_green + (1 if action < remainder else 0)
            for action in range(action_count)
        ]

    def _enter_cv_absence_fallback(self, tls_id, current_time):
        action_count = self.net_info[tls_id]['phase_num'] // 2
        current_phase = int(
            traci.trafficlight.getPhase(tls_id))
        if current_phase % 2 == 1:
            current_action = (
                (current_phase + 1) // 2) % action_count
            traci.trafficlight.setPhase(
                tls_id, current_action * 2)
        else:
            current_action = (
                current_phase // 2) % action_count
        # Restart the current green so the fallback lasts exactly one complete
        # cycle regardless of how long that phase had already been active.
        traci.trafficlight.setPhase(
            tls_id, current_action * 2)
        durations = self._fallback_green_durations(tls_id)
        self.cv_fallback_state[tls_id] = {
            'active': True,
            'stage': 'green',
            'action': current_action,
            'target_action': current_action,
            'start_action': current_action,
            'action_count': action_count,
            'completed_green_count': 0,
            'green_durations': durations,
            'next_event_time': (
                current_time + durations[current_action]),
            'yellow_end_time': None,
            'cycle_start_time': current_time,
            'cycle_end_time': (
                current_time + self.fallback_cycle_seconds),
            'reason': (
                'no_cv' if self._intersection_cv_count(tls_id) <= 0
                else 'no_serviceable_positive_pressure'),
        }
        self.g_max_flag[tls_id] = 0
        self.phase_switch_pointer[tls_id] = current_time
        return current_action

    def _exit_cv_absence_fallback(self, tls_id):
        self.cv_fallback_state.pop(tls_id, None)

    def _resolve_cv_absence_fallback(
            self, tls_id, proposed_action, decision_now):
        if not self.cv_absence_fixed_time_fallback:
            return None

        current_time = traci.simulation.getTime()
        state = self.cv_fallback_state.get(tls_id)
        if state is None:
            if not decision_now or not self._needs_fixed_cycle(tls_id):
                return None
            return self._enter_cv_absence_fallback(
                tls_id, current_time)

        if state['stage'] == 'green':
            if decision_now and current_time >= state['next_event_time']:
                state['completed_green_count'] += 1
                target_action = (
                    state['action'] + 1) % state['action_count']
                state['stage'] = 'yellow'
                state['target_action'] = target_action
                state['yellow_end_time'] = (
                    current_time + self._yellow)
                return target_action
            return state['action']

        if (decision_now and
                current_time >= state['yellow_end_time'] and
                state['completed_green_count'] >= state['action_count']):
            target_phase = state['target_action'] * 2
            traci.trafficlight.setPhase(tls_id, target_phase)
            self.phase_switch_pointer[tls_id] = current_time
            self._exit_cv_absence_fallback(tls_id)
            if self._needs_fixed_cycle(tls_id):
                return self._enter_cv_absence_fallback(
                    tls_id, current_time)
            return proposed_action

        return state['target_action']

    def _finalize_cv_absence_fallback(self):
        if not self.cv_absence_fixed_time_fallback:
            return

        current_time = traci.simulation.getTime()
        for tls_id, state in list(self.cv_fallback_state.items()):
            if state['stage'] == 'yellow':
                current_phase = traci.trafficlight.getPhase(tls_id)
                target_phase = state['target_action'] * 2
                if (current_time >= state['yellow_end_time'] and
                        current_phase == target_phase):
                    # SUMO may enter the target green automatically at the end
                    # of yellow. Restart it here so its full fallback green
                    # duration is measured from this exact timestamp.
                    traci.trafficlight.setPhase(
                        tls_id, target_phase)
                    self.phase_switch_pointer[tls_id] = current_time
                    if (state['completed_green_count'] >=
                            state['action_count']):
                        self._exit_cv_absence_fallback(tls_id)
                        self.decision_step[tls_id] = current_time
                        self.next_decision_flag[tls_id] = 1
                        continue

                    state['stage'] = 'green'
                    state['action'] = state['target_action']
                    green_start = current_time
                    # After a yellow transition, the base environment exposes
                    # the next decision flag one simulation step later. Offset
                    # that boundary so the realized green remains the intended
                    # duration instead of gaining one extra second.
                    green_duration = max(
                        1, state['green_durations'][state['action']] - 1)
                    state['next_event_time'] = (
                        green_start + green_duration)
                    state['yellow_end_time'] = None

            if state['stage'] == 'green':
                self.decision_step[tls_id] = max(
                    current_time, state['next_event_time'])
            else:
                self.decision_step[tls_id] = max(
                    current_time, state['yellow_end_time'])
            self.next_decision_flag[tls_id] = int(
                self.decision_step[tls_id] == current_time)

    def _simulate(self, steps_todo=1):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            self._update_cv_vehicle_states()

        return self._step

    def _is_cv_vehicle(self, veh_id):
        if veh_id in self.cv_vehicle_ids:
            return True
        if veh_id in self.nv_vehicle_ids:
            return False

        if self.cv_penetration_rate >= 1.0:
            self.cv_vehicle_ids.add(veh_id)
            return True
        if self.cv_penetration_rate <= 0.0:
            self.nv_vehicle_ids.add(veh_id)
            return False

        if self.cv_random.random() <= self.cv_penetration_rate:
            self.cv_vehicle_ids.add(veh_id)
            return True

        self.nv_vehicle_ids.add(veh_id)
        return False

    def _update_cv_vehicle_states(self):
        self._ensure_cvmp_network_info()
        current_time = traci.simulation.getTime()

        for veh_id in traci.simulation.getDepartedIDList():
            if self._is_cv_vehicle(veh_id):
                self.active_cv_ids.add(veh_id)
                self._cache_vehicle_route(veh_id)

        for veh_id in traci.simulation.getArrivedIDList():
            self._remove_cv_vehicle(veh_id)

        for veh_id in list(self.active_cv_ids):
            try:
                lane_id = traci.vehicle.getLaneID(veh_id)
            except traci.TraCIException:
                self._remove_cv_vehicle(veh_id)
                continue

            self._update_cv_vehicle_movement(veh_id, lane_id, current_time)

    def _cache_vehicle_route(self, veh_id):
        route = self.vehicle_route.get(veh_id)
        if route is None:
            route = tuple(traci.vehicle.getRoute(veh_id))
            self.vehicle_route[veh_id] = route
        return route

    def _remove_vehicle_contribution(self, veh_id):
        movement_key = self.vehicle_movement_key.get(veh_id)
        if movement_key is None:
            return

        self.movement_states[movement_key] -= self.vehicle_tau.get(veh_id, 0.0)
        self.movement_counts[movement_key] -= 1

        if self.movement_states[movement_key] < 1e-9:
            self.movement_states[movement_key] = 0.0
        if self.movement_counts[movement_key] < 0:
            self.movement_counts[movement_key] = 0

        self.vehicle_movement_key[veh_id] = None
        self.vehicle_tau[veh_id] = 0.0

    def _remove_cv_vehicle(self, veh_id):
        self._remove_vehicle_contribution(veh_id)
        self.active_cv_ids.discard(veh_id)
        self.vehicle_lane.pop(veh_id, None)
        self.vehicle_edge.pop(veh_id, None)
        self.vehicle_edge_enter_time.pop(veh_id, None)
        self.vehicle_route.pop(veh_id, None)
        self.vehicle_route_index.pop(veh_id, None)
        self.vehicle_movement_key.pop(veh_id, None)
        self.vehicle_tau.pop(veh_id, None)

    def _get_edge_free_flow_time(self, edge_id, lane_id):
        if edge_id not in self.edge_free_flow_time:
            lane_length = max(traci.lane.getLength(lane_id), 1e-6)
            lane_speed = max(traci.lane.getMaxSpeed(lane_id), 1e-6)
            self.edge_free_flow_time[edge_id] = max(
                lane_length / lane_speed, 1e-6)
        return self.edge_free_flow_time[edge_id]

    def _get_vehicle_next_edge(self, veh_id, current_edge):
        route = self._cache_vehicle_route(veh_id)
        if not route:
            return None

        route_index = self.vehicle_route_index.get(veh_id)
        if route_index is not None and 0 <= route_index < len(route) and route[route_index] == current_edge:
            next_index = route_index + 1
            return route[next_index] if next_index < len(route) else None

        try:
            edge_index = route.index(current_edge)
        except ValueError:
            route = tuple(traci.vehicle.getRoute(veh_id))
            self.vehicle_route[veh_id] = route
            try:
                edge_index = route.index(current_edge)
            except ValueError:
                return None

        self.vehicle_route_index[veh_id] = edge_index
        next_index = edge_index + 1
        return route[next_index] if next_index < len(route) else None

    def _get_cv_normalized_edge_travel_time(self, veh_id, lane_id,
                                            current_time):
        edge_id = self._lane_to_edge(lane_id)
        if self.vehicle_edge.get(veh_id) != edge_id:
            self.vehicle_edge[veh_id] = edge_id
            self.vehicle_edge_enter_time[veh_id] = current_time

        enter_time = self.vehicle_edge_enter_time.get(veh_id, current_time)
        free_flow_time = self._get_edge_free_flow_time(edge_id, lane_id)
        return max(0.0, (current_time - enter_time) / free_flow_time)

    def _update_cv_vehicle_movement(self, veh_id, lane_id, current_time):
        if not lane_id or lane_id.startswith(':') or lane_id not in self.movement_by_in_lane_out_edge:
            self._remove_vehicle_contribution(veh_id)
            self.vehicle_lane[veh_id] = lane_id
            return

        current_edge = self._lane_to_edge(lane_id)
        self._get_cv_normalized_edge_travel_time(
            veh_id, lane_id, current_time)
        next_edge = self._get_vehicle_next_edge(veh_id, current_edge)
        movement = self.movement_by_in_lane_out_edge[lane_id].get(next_edge)

        if movement is None:
            self._remove_vehicle_contribution(veh_id)
            self.vehicle_lane[veh_id] = lane_id
            return

        movement_key = movement['key']
        lane_changed = self.vehicle_lane.get(veh_id) != lane_id
        movement_changed = self.vehicle_movement_key.get(veh_id) != movement_key

        if lane_changed or movement_changed:
            self._remove_vehicle_contribution(veh_id)
            self.vehicle_lane[veh_id] = lane_id
            self.vehicle_movement_key[veh_id] = movement_key
            new_tau = self._get_cv_normalized_edge_travel_time(
                veh_id, lane_id, current_time)
            self.vehicle_tau[veh_id] = new_tau
            self.movement_states[movement_key] += new_tau
            self.movement_counts[movement_key] += 1
            return

        old_tau = self.vehicle_tau.get(veh_id, 0.0)
        new_tau = self._get_cv_normalized_edge_travel_time(
            veh_id, lane_id, current_time)
        self.movement_states[movement_key] += new_tau - old_tau
        self.vehicle_tau[veh_id] = new_tau

    def _collect_cv_lane_travel_time(self, lane_id):
        current_time = traci.simulation.getTime()
        cv_travel_time = 0.0

        for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
            if not self._is_cv_vehicle(veh_id):
                continue

            normalized_travel_time = (
                self._get_cv_normalized_edge_travel_time(
                    veh_id, lane_id, current_time))
            cv_travel_time += normalized_travel_time

        return cv_travel_time

    def _ensure_cvmp_network_info(self):
        if self.cvmp_network_ready:
            return

        self.tls_action_movements = {}
        self.movements_by_in_lane = defaultdict(list)
        self.downstream_count_movements_by_in_edge = defaultdict(list)
        self.downstream_state_movements_by_in_edge = defaultdict(list)
        self.movement_by_in_lane_out_edge = defaultdict(dict)
        self.all_cvmp_movements = []
        movement_keys = set()

        for tls_id in self.net_info.keys():
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            phases = program.phases

            tls_movements = []
            for link_index, link_group in enumerate(controlled_links):
                for link in link_group:
                    in_lane, out_lane, _ = link
                    if not in_lane or not out_lane:
                        continue
                    if in_lane.startswith(':') or out_lane.startswith(':'):
                        continue

                    movement_key = (in_lane, self._lane_to_edge(out_lane))
                    movement = {
                        'key': movement_key,
                        'tls_id': tls_id,
                        'link_index': link_index,
                        'in_lane': in_lane,
                        'out_lane': out_lane,
                        'in_edge': self._lane_to_edge(in_lane),
                        'out_edge': self._lane_to_edge(out_lane),
                    }
                    tls_movements.append(movement)

                    if movement_key not in movement_keys:
                        movement_keys.add(movement_key)
                        self.all_cvmp_movements.append(movement)
                        self.movements_by_in_lane[in_lane].append(movement)
                        self.downstream_count_movements_by_in_edge[movement['in_edge']].append(movement)
                        self.movement_by_in_lane_out_edge[in_lane][movement['out_edge']] = movement
                        # Keep the same no-right-turn pressure scope as the original Q-MP code,
                        # while still counting right-turn CVs in the denominator of turning ratios.
                        if self._lane_turn(in_lane) is not None:
                            self.downstream_state_movements_by_in_edge[movement['in_edge']].append(movement)

            action_movements = []
            for action_idx in range(self.net_info[tls_id]['phase_num'] // 2):
                phase_idx = action_idx * 2
                signal_state = phases[phase_idx].state
                phase_movements = [
                    movement for movement in tls_movements
                    if movement['link_index'] < len(signal_state)
                    and signal_state[movement['link_index']] in ('G', 'g')
                    and self._lane_turn(movement['in_lane']) is not None
                ]
                action_movements.append(phase_movements)

            self.tls_action_movements[tls_id] = action_movements
        
        self.cvmp_network_ready = True

    def _collect_cv_movement_states(self):
        self._ensure_cvmp_network_info()
        return self.movement_states, self.movement_counts

    def _dynamic_downstream_state(self, movement, movement_states,
                                  movement_counts):
        current_time = traci.simulation.getTime()
        if self.downstream_state_cache_time != current_time:
            self.downstream_state_cache_time = current_time
            self.downstream_state_cache = {}

        movement_key = movement['key']
        if movement_key in self.downstream_state_cache:
            return self.downstream_state_cache[movement_key]

        count_movements = self.downstream_count_movements_by_in_edge.get(movement['out_edge'], [])
        state_movements = self.downstream_state_movements_by_in_edge.get(movement['out_edge'], [])
        if not count_movements or not state_movements:
            self.downstream_state_cache[movement_key] = 0.0
            return 0.0

        # Use the same all-vehicle, edge-level turning ratios as MP. The
        # movement states being weighted remain CV normalized travel times.
        _, all_vehicle_counts = self._collect_mp_movement_states()
        total_vehicle_count = sum(
            all_vehicle_counts[downstream['key']]
            for downstream in count_movements)
        if total_vehicle_count <= 0:
            self.downstream_state_cache[movement_key] = 0.0
            return 0.0

        downstream_state = 0.0
        # print("movement_key:", movement_key)
        # print("state_movements:", state_movements)
        for downstream in state_movements:
            # print("downstream:", downstream)
            turning_ratio = (
                all_vehicle_counts[downstream['key']] /
                total_vehicle_count)
            # print("turning_ratio:", turning_ratio)
            downstream_state += turning_ratio * movement_states[downstream['key']]
        
        self.downstream_state_cache[movement_key] = downstream_state
        return downstream_state

    def _movement_pressure(self, movement, movement_states, movement_counts):
        in_state = movement_states[movement['key']]
        downstream_state = self._dynamic_downstream_state(movement,
                                                          movement_states,
                                                          movement_counts)
        return max(0.0, in_state - downstream_state)

    # state function candidates
    def _collect_pressure(self, tls_id):
        self._ensure_cvmp_network_info()
        current_time = traci.simulation.getTime()
        if self.pressure_cache_time != current_time:
            self.pressure_cache_time = current_time
            self.pressure_cache = {}

        if tls_id in self.pressure_cache:
            return self.pressure_cache[tls_id]

        movement_states, movement_counts = self._collect_cv_movement_states()

        pressure = []
        raw_pressure = []
        serviceability = []
        for phase_movements in self.tls_action_movements[tls_id]:
            phase_pressure = 0.0
            movement_pressure_by_key = {}
            for movement in phase_movements:
                movement_pressure = self._movement_pressure(
                    movement, movement_states, movement_counts)
                movement_pressure_by_key[movement['key']] = (
                    movement_pressure)
                phase_pressure += movement_pressure

            phase_pressure *= self.saturation_flow_rate
            phase_serviceable = (
                phase_pressure > 1e-9 and
                self._phase_heads_can_pass(
                    phase_movements, movement_pressure_by_key))
            raw_pressure.append(phase_pressure)
            serviceability.append(phase_serviceable)
            pressure.append(
                phase_pressure if phase_serviceable else 0.0)
        
        self.cvmp_raw_pressure[tls_id] = raw_pressure
        self.cvmp_phase_serviceability[tls_id] = serviceability
        self.pressure_cache[tls_id] = pressure
        return pressure

    def _empty_pressure(self, tls_id):
        return [0.0] * (self.net_info[tls_id]['phase_num'] // 2)
