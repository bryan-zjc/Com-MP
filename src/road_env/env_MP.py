# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:44:00 2024

"""

import os
import traci
# import wandb
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
import bs4
import random
import copy
import time
import xml.etree.ElementTree as ET

'''
Intersec_Env for rl offline data collection, training, and fine-tuning
'''

class Intersec_Env:
    def __init__(self, rou_file, net_file, num_agent, sumo_cmd, max_steps, seed):
        # sumo config
        # self.MP_PR = MP_PR
        self._max_steps = max_steps
        self._sumo_cmd = sumo_cmd
        # self.t_sequence = t_sequence
        # self._generator = generator

        # signal control parameters config
        self._yellow = 3
        self._green_min = 15
        self._green_max = 40
        self._decision_interval = 10
        
        # agent config
        self.num_agent = num_agent

        # self.control_input = [0 for _ in range(num_agent)]
        self.net_info = None # a dict to store the basic intersection info
        self.net_links = [] # a list to store the edge id in the network (except the out links at the border)
        
        # code config
        # self.wandb = wandb
        
        # random seed
        self.seed = seed

        # MP pressure config
        self.saturation_flow_rate = 1800
        self.mp_network_ready = False
        self.mp_pressure_cache_time = None
        self.mp_pressure_cache = {}
        self.mp_state_cache_time = None
        self.mp_movement_states = defaultdict(float)
        self.mp_movement_counts = defaultdict(int)
        self.mp_downstream_state_cache_time = None
        self.mp_downstream_state_cache = {}
        
    def _lane_turn(self, lane_id):
        idx = lane_id.rsplit('_', 1)[-1]
        if idx == '1':
            return 's'
        if idx == '2':
            return 'l'
        return None  # right or unknown

    def _lane_dir(self, lane_id, incoming=True):
        pts = traci.lane.getShape(lane_id)
        if len(pts) < 2:
            return None
        if incoming:
            x0, y0 = pts[-1]   # near junction
            x1, y1 = pts[0]    # upstream
        else:
            x0, y0 = pts[0]    # near junction
            x1, y1 = pts[-1]   # downstream
        dx, dy = x1 - x0, y1 - y0
        if abs(dx) >= abs(dy):
            return 'E' if dx > 0 else 'W'
        return 'N' if dy > 0 else 'S'

    def _sorted_no_right(self, lanes, incoming=True):
        slots = {}
        for lane in lanes:
            turn = self._lane_turn(lane)
            if turn is None:
                continue
            d = self._lane_dir(lane, incoming=incoming)
            if d is None:
                continue
            slots[(d, turn)] = lane

        order = [('E', 's'), ('E', 'l'),
                ('S', 's'), ('S', 'l'),
                ('W', 's'), ('W', 'l'),
                ('N', 's'), ('N', 'l')]
        ordered = [slots[k] for k in order if k in slots]

        fallback = [l for l in lanes if self._lane_turn(l) is not None]
        return ordered if len(ordered) == len(fallback) == 8 else fallback

    def _lane_to_edge(self, lane_id):
        return lane_id.rsplit('_', 1)[0]


    def reset(self, MPtripinfo_path):
        self.mp_network_ready = False
        self.mp_pressure_cache_time = None
        self.mp_pressure_cache = {}
        self.mp_state_cache_time = None
        self.mp_movement_states = defaultdict(float)
        self.mp_movement_counts = defaultdict(int)
        self.mp_downstream_state_cache_time = None
        self.mp_downstream_state_cache = {}

        # start sumo with traci
        sumo_cmd1 = self._sumo_cmd + ["--tripinfo-output", MPtripinfo_path]
        traci.start(sumo_cmd1) 


        # retrieve the basic inter info
        if not self.net_info:
            tls_ids = traci.trafficlight.getIDList()
            self.net_info = defaultdict(dict)
            for tls_id in tls_ids:
                phase_num = len((traci.trafficlight.getAllProgramLogics(tls_id)[0]).phases)
                lanes_all, lanes_in, lanes_out, lanes_connect, edge_in = self._read_in_and_out(traci.trafficlight.getControlledLinks(tls_id))
                self.net_info[tls_id]['phase_num'] = phase_num
                self.net_info[tls_id]['lanes_all'] = lanes_all
                self.net_info[tls_id]['lanes_in'] = lanes_in
                self.net_info[tls_id]['lanes_out'] = lanes_out
                self.net_info[tls_id]['lanes_connect'] = lanes_connect
                self.net_info[tls_id]['edges_in'] = edge_in
                self.net_info[tls_id]['phase2lane'], self.net_info[tls_id]['lane2phase'] = self._read_phase2lane(tls_id)
                
                # 进口道顺序：东进口道直行、东进口道左转、南进口道直行、南进口道左转、西进口道直行、西进口道左转、北进口道直行、北进口道左转
                self.net_info[tls_id]['lanes_in_no_right'] = self._sorted_no_right(lanes_in, incoming=True)
                # 出口道顺序：西出口道直行，南出口道左转，北出口道直行，西出口道左转，东出口道直行，北出口道左转，南出口道直行，东出口道左转
                self.net_info[tls_id]['lanes_out_no_right'] = self._sorted_no_right(lanes_out, incoming=False)
                # print(self.net_info[tls_id]['lanes_in_no_right'])

                # self.net_info[tls_id]['lanes_in_no_right'] = []
                # self.net_info[tls_id]['lanes_out_no_right'] = []
                # for lane in lanes_in:
                #     if lane.split('_')[-1] != '0':
                #         self.net_info[tls_id]['lanes_in_no_right'].append(lane)
                # for lane in lanes_out:
                #     if lane.split('_')[-1] != '0':
                #         self.net_info[tls_id]['lanes_out_no_right'].append(lane)
                
                self.net_links+=edge_in
        
        # for tls_id in tls_ids:      
        #     print(tls_id)
        #     print(self.net_info[tls_id]['lanes_in_no_right'])
        #     print(self.net_info[tls_id]['lanes_out_no_right'])
        
        # initialize the env
        self.phase_switch_pointer = {} # save the start time of current phase
        self._policy_dict = {} # save the policy info
        
        self.g_max_flag = {}
        
        
        for tls_id in self.net_info.keys():
            traci.trafficlight.setPhase(tls_id, 0)
            self.phase_switch_pointer[tls_id] = 0
            self._policy_dict[tls_id] = []
            self.g_max_flag[tls_id] = 0
        
        self._step = 0

        self._simulate()
        self.decision_step = {}
        
        for tls_id in self.net_info.keys():
            self.decision_step[tls_id] = self._step + self._green_min
        
        total_pressure = {}
        for tls_id in self.net_info.keys():
            total_pressure[tls_id] = self._collect_pressure(tls_id)
        
        return total_pressure, self.net_info


    def close(self):
        traci.close()
        # print("----- connection to sumo closed")

    def fixed_time_warm_start(self, warm_start_steps):
        warm_start_steps = max(0, int(warm_start_steps))
        if warm_start_steps <= self._step:
            decision_flag = {tls_id: 1 for tls_id in self.net_info.keys()}
            total_pressure = {
                tls_id: self._collect_pressure(tls_id)
                for tls_id in self.net_info.keys()
            }
            return total_pressure, decision_flag

        steps_todo = min(warm_start_steps - self._step, self._max_steps - self._step)
        if steps_todo > 0:
            self._simulate(steps_todo)

        current_time = traci.simulation.getTime()
        decision_flag = {}
        total_pressure = {}
        self.next_decision_flag = {}

        for tls_id in self.net_info.keys():
            self.g_max_flag[tls_id] = 0
            self.phase_switch_pointer[tls_id] = current_time
            self.decision_step[tls_id] = current_time
            decision_flag[tls_id] = 1
            self.next_decision_flag[tls_id] = 1
            total_pressure[tls_id] = self._collect_pressure(tls_id)

        return total_pressure, decision_flag


    def step(self, action):
        # self.control_input = action
        # print('self.g_max_flag: ', self.g_max_flag)
        self.control_input = {}
        for tls_id in self.net_info.keys():
            if self.g_max_flag[tls_id] == 1:
                current_phase = traci.trafficlight.getPhase(tls_id)
                self.control_input[tls_id] = ((current_phase + 1) % self.net_info[tls_id]['phase_num'])/2 #人为修改下一次的执行相位，此时相位为黄灯
            else:
                self.control_input[tls_id] = action[tls_id] # apply action to the signal
        # print('control input: ', self.control_input)
        self._update_env() # update the env
        
        total_pressure = {}
        dones = {}
        for tls_id in self.net_info.keys():
            total_pressure[tls_id] = self._collect_pressure(tls_id)
            dones[tls_id]=[self._step >= self._max_steps-15]
            
        return total_pressure, dones, self.next_decision_flag


    def _count_waiting_vehicles(self, lane_id, v_th):
        veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        return sum(1 for vid in veh_ids if traci.vehicle.getSpeed(vid) < v_th)


    # state function candidates
    def _collect_pressure(self, tls_id):
        self._ensure_mp_network_info()
        current_time = traci.simulation.getTime()
        if self.mp_pressure_cache_time != current_time:
            self.mp_pressure_cache_time = current_time
            self.mp_pressure_cache = {}

        if tls_id in self.mp_pressure_cache:
            return self.mp_pressure_cache[tls_id]

        movement_states, movement_counts = self._collect_mp_movement_states()

        pressure = []
        for phase_movements in self.mp_tls_action_movements[tls_id]:
            phase_pressure = 0.0
            for movement in phase_movements:
                phase_pressure += self._mp_movement_pressure(movement,
                                                             movement_states,
                                                             movement_counts)
            pressure.append(phase_pressure * self.saturation_flow_rate)

        self.mp_pressure_cache[tls_id] = pressure
        return pressure

    def _ensure_mp_network_info(self):
        if self.mp_network_ready:
            return

        self.mp_tls_action_movements = {}
        self.mp_downstream_count_movements_by_in_edge = defaultdict(list)
        self.mp_downstream_state_movements_by_in_edge = defaultdict(list)
        self.mp_all_movements = []
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
                        self.mp_all_movements.append(movement)
                        self.mp_downstream_count_movements_by_in_edge[movement['in_edge']].append(movement)
                        # Right turns are counted in the denominator of turning ratios,
                        # but keep the original no-right-turn MP pressure scope.
                        if self._lane_turn(in_lane) is not None:
                            self.mp_downstream_state_movements_by_in_edge[movement['in_edge']].append(movement)

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

            self.mp_tls_action_movements[tls_id] = action_movements

        self.mp_network_ready = True

    def _collect_mp_movement_states(self):
        self._ensure_mp_network_info()
        current_time = traci.simulation.getTime()
        if self.mp_state_cache_time == current_time:
            return self.mp_movement_states, self.mp_movement_counts

        self.mp_state_cache_time = current_time
        self.mp_movement_states = defaultdict(float)
        self.mp_movement_counts = defaultdict(int)

        for movement in self.mp_all_movements:
            vehicle_count = traci.lane.getLastStepVehicleNumber(movement['in_lane'])
            self.mp_movement_states[movement['key']] = vehicle_count
            self.mp_movement_counts[movement['key']] = vehicle_count

        return self.mp_movement_states, self.mp_movement_counts

    def _mp_dynamic_downstream_state(self, movement, movement_states, movement_counts):
        current_time = traci.simulation.getTime()
        if self.mp_downstream_state_cache_time != current_time:
            self.mp_downstream_state_cache_time = current_time
            self.mp_downstream_state_cache = {}

        movement_key = movement['key']
        if movement_key in self.mp_downstream_state_cache:
            return self.mp_downstream_state_cache[movement_key]

        count_movements = self.mp_downstream_count_movements_by_in_edge.get(movement['out_edge'], [])
        state_movements = self.mp_downstream_state_movements_by_in_edge.get(movement['out_edge'], [])
        if not count_movements or not state_movements:
            self.mp_downstream_state_cache[movement_key] = 0.0
            return 0.0

        total_vehicle_count = sum(movement_counts[downstream['key']]
                                  for downstream in count_movements)
        if total_vehicle_count <= 0:
            self.mp_downstream_state_cache[movement_key] = 0.0
            return 0.0

        downstream_state = 0.0
        for downstream in state_movements:
            turning_ratio = movement_counts[downstream['key']] / total_vehicle_count
            downstream_state += turning_ratio * movement_states[downstream['key']]

        self.mp_downstream_state_cache[movement_key] = downstream_state
        return downstream_state

    def _mp_movement_pressure(self, movement, movement_states, movement_counts):
        in_state = movement_states[movement['key']]
        downstream_state = self._mp_dynamic_downstream_state(movement,
                                                             movement_states,
                                                             movement_counts)
        return in_state - downstream_state
    


    # network information reading
    def _read_in_and_out(self, controlledlinks):
        lanes_all = []
        lanes_in = []
        lanes_out = []
        lanes_connect = []
        edges_in = []
        for sublist in controlledlinks:
            for item in sublist:
                lanes_in.append(item[0])
                lanes_out.append(item[1])
                lanes_connect.append(item[2])
        lanes_all = (lanes_in + lanes_out)
        # lanes_in is sorted by north-east-south-west
        
        # lanes_in = list(set(lanes_in))
        # lanes_out = list(set(lanes_out))
        # lanes_connect = list(set(lanes_connect))
        # 排序
        # lanes_all.sort()
        # lanes_in.sort()
        # lanes_out.sort()
        # lanes_connect.sort()
        
        for lane in lanes_in:
            edges_in.append(lane[:-2])
        edges_in = list(set(edges_in))
        edges_in.sort()
        
        return tuple(lanes_all), tuple(lanes_in), tuple(lanes_out), tuple(lanes_connect), tuple(edges_in)
    
    
    def _read_phase2lane(self, tls_id):
        phase2lane = {}
        lane2phase = {}
        for phase_idx in range(self.net_info[tls_id]['phase_num']):
            signal_state = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases[phase_idx].state
            # extract the index of G and g in the signal state 并集
            G_idx = [i for i, x in enumerate(signal_state) if x == 'G' or x == 'g']
            list = []
            for idx in G_idx:
                list.append(traci.trafficlight.getControlledLinks(tls_id)[idx][0][0])
                phase2lane[phase_idx] = set(list)
                for edge in list:
                    lane2phase[edge] = phase_idx
        
        return phase2lane, lane2phase
    

    def _update_env(self):
        phases = {}
        switch_flag = {}
        decision_flag = {}
        self.next_decision_flag = {}
        for tls_id in self.net_info.keys():
            phases[tls_id] = traci.trafficlight.getPhase(tls_id)
            # print( traci.simulation.getTime())
            if self.decision_step[tls_id] == traci.simulation.getTime():
                decision_flag[tls_id] = 1
            else:
                decision_flag[tls_id] = 0
            if self.control_input[tls_id]*2 != phases[tls_id]:
                switch_flag[tls_id] = 1
            else:
                switch_flag[tls_id] = 0
        # print('g_max_flag: ', self.g_max_flag)
        # print('current step: ', self._step)
        # print('decision flag:', decision_flag)
        # print('switch flag: ', switch_flag)
        # print('control input: ', self.control_input)
        # print('decision_step: ', self.decision_step)
        # print('phase_switch_pointer: ', self.phase_switch_pointer)
        # print('-------------------------------------------------')
        # set the action to signal switch a large step when the phase is in yellow
        for tls_id in self.net_info.keys():
            if decision_flag[tls_id] == 1:
                if switch_flag[tls_id] == 1:
                    # exectute yellow phase
                    self._set_next_phase(tls_id)
                    self.decision_step[tls_id] = traci.simulation.getTime()+self._yellow+self._green_min
                else:
                    if (traci.simulation.getTime() >=
                            self.phase_switch_pointer[tls_id] +
                            self._green_max):
                        self.g_max_flag[tls_id] = 1
                        self._set_next_phase(tls_id)
                        self.decision_step[tls_id] = traci.simulation.getTime()+self._yellow+self._green_min
                    else:
                        self._set_control_phase(tls_id)
                        self.decision_step[tls_id] = min(
                            traci.simulation.getTime() +
                            self._decision_interval,
                            self.phase_switch_pointer[tls_id] +
                            self._green_max)
            else: #decision_flag = 0
                if traci.simulation.getTime() == self.phase_switch_pointer[tls_id]+self._yellow and traci.simulation.getTime()>3 and phases[tls_id]%2 == 1:
                    self._set_control_phase(tls_id)
                    self.g_max_flag[tls_id] = 0

        
        self._simulate(steps_todo=1)
        # print('decision_step:',self.decision_step)
        
        for tls_id in self.net_info.keys():
            if self.decision_step[tls_id] == traci.simulation.getTime():
                self.next_decision_flag[tls_id] = 1
            else:
                self.next_decision_flag[tls_id] = 0
        
        
    def _simulate(self, steps_todo=1):
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step
        while steps_todo > 0:
            phases = {}
            for tls_id in self.net_info.keys():
                phases[tls_id] = traci.trafficlight.getPhase(tls_id)
                # print('phases', phases[tls_id])

            traci.simulationStep()  # simulate 1 step in sumo
            
            # collect data to train reward estimator
            next_phases = {}
            for tls_id in self.net_info.keys():
                next_phases[tls_id] = traci.trafficlight.getPhase(tls_id)
                # print('next_phases', next_phases[tls_id])
                # self._collect_policy_info(tls_id, next_phases[tls_id])  # save the policy info
                # if next_phases[tls_id] != phases[tls_id]:
                #     self.phase_switch_pointer[tls_id] = traci.simulation.getTime() -1
            
            self._step += 1 # update the step counter
            steps_todo -= 1
            
            # self._veh_delay_collect() # collect the delay info of the vehicles
            
        return self._step

    
    def _set_next_phase(self, tls_id):
        current_phase = traci.trafficlight.getPhase(tls_id)
        next_phase = (current_phase + 1) % self.net_info[tls_id]['phase_num']
        # print(f'current phase: {current_phase}, next phase: {next_phase}')
        traci.trafficlight.setPhase(tls_id, next_phase)
        
        self.phase_switch_pointer[tls_id] = traci.simulation.getTime()
        # print('time to change the phase', traci.simulation.getTime())
    
    def _set_control_phase(self,tls_id):
        current_phase = traci.trafficlight.getPhase(tls_id)
        next_phase = self.control_input[tls_id]*2
        traci.trafficlight.setPhase(tls_id, next_phase)
        # print('tls_id: ', tls_id)
        # print('current phase: ', current_phase)
        # print('next phase: ', next_phase)
        if current_phase != next_phase:
            self.phase_switch_pointer[tls_id] = traci.simulation.getTime()
    

    


    

        
    
