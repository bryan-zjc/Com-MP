from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .sumo_bootstrap import ensure_sumo_python_tools
ensure_sumo_python_tools()
import traci
from .config import CVMPConfig
from .topology import CVRouteTopology, MovementDef, PhysicalEdgeGraph

@dataclass(slots=True)
class VehicleSnapshot:
    vehicle_id: str
    logical_link: str
    movement_id: str
    link_entry_time_s: int
    normalized_link_time: float
    is_uncontrolled_bypass: bool = False

@dataclass(slots=True)
class PhaseEvaluation:
    phase_id: int
    score: float
    pressure_score: float
    service_factor: float
    movement_details: dict[str, dict[str, float]]

@dataclass(slots=True)
class ServiceIntervalObservation:
    served: bool
    waiting_connected: int
    passed_connected: int

@dataclass(frozen=True, slots=True)
class ServiceFailureTrigger:
    movement_id: str
    waiting_prev2: tuple[int, int]
    passed_prev2: tuple[int, int]

class CVMPTracker:
    """Passive connected-vehicle state tracker. It reads each vehicle's fixed SUMO route only to identify its current movement; it never changes a route."""
    def __init__(self, topology: CVRouteTopology, graph: PhysicalEdgeGraph) -> None:
        self.topology = topology
        self.graph = graph
        self.active_connected: set[str] = set()
        self.current_link: dict[str, str] = {}
        self.current_movement: dict[str, str] = {}
        self.current_is_bypass: dict[str, bool] = {}
        self.link_entry_time: dict[str, int] = {}
        self.last_road: dict[str, str] = {}
        self.mapping_failures_total = 0
        self.last_raw_w: dict[str, float] = {m: 0.0 for m in topology.movements}

    def _movement_from_next_tls_for_link(self, next_tls, link_id: str) -> str | None:
        for item in next_tls or ():
            if len(item) < 2:
                continue
            try:
                key = (str(item[0]), int(item[1]))
            except Exception:
                continue
            movement = self.topology.movement_by_tls_link_index.get(key)
            if movement is not None and self.topology.movements[movement].source_link == link_id:
                return movement
        return None

    @staticmethod
    def _route_position(route: Sequence[str], route_index: int, road: str) -> int | None:
        lo = min(max(int(route_index), 0), max(len(route) - 1, 0))
        for j in range(max(0, lo - 1), len(route)):
            if route[j] == road:
                return j
        return None

    def _movement_on_fixed_link(self, vehicle_id: str, link_id: str, road: str, vehicle_api) -> tuple[str | None, bool]:
        try:
            route = tuple(str(e) for e in vehicle_api.getRoute(vehicle_id))
            route_index = int(vehicle_api.getRouteIndex(vehicle_id))
        except Exception:
            return None, False
        pos = self._route_position(route, route_index, road)
        if pos is None:
            return None, False
        link = self.topology.links[link_id]
        link_edges = set(link.edges)
        if route[pos] not in link_edges:
            return None, False
        for j in range(pos, len(route)):
            if route[j] not in link_edges:
                break
            if j + 1 >= len(route):
                return None, False
            nxt = route[j + 1]
            if nxt in link_edges:
                continue
            movement_id = self.topology.routing_movement_by_edge_pair.get((route[j], nxt))
            if movement_id is None:
                return None, False
            return movement_id, not self.topology.is_controlled_routing_movement(movement_id)
        return None, False

    def _clear_assignment(self, vehicle_id: str) -> None:
        self.current_link.pop(vehicle_id, None)
        self.current_movement.pop(vehicle_id, None)
        self.current_is_bypass.pop(vehicle_id, None)
        self.link_entry_time.pop(vehicle_id, None)

    def _next_normal_route_edge(self, vehicle_id: str, vehicle_api) -> str | None:
        try:
            route = tuple(str(e) for e in vehicle_api.getRoute(vehicle_id))
            idx = max(int(vehicle_api.getRouteIndex(vehicle_id)), 0)
        except Exception:
            return None
        for j in range(min(idx, len(route)), len(route)):
            edge = route[j]
            if edge in self.graph.edge_length_m:
                return edge
        return None

    def _update_vehicle(self, vehicle_id: str, road: str, now: int, vehicle_api, *, next_tls=None) -> None:
        old_road = self.last_road.get(vehicle_id, "")
        old_link = self.current_link.get(vehicle_id)
        if old_road and old_road != road and old_link is not None and old_road == self.topology.links[old_link].terminal_edge:
            self._clear_assignment(vehicle_id)
            old_link = None
        self.last_road[vehicle_id] = road
        if road not in self.graph.edge_length_m:
            old_link = self.current_link.get(vehicle_id)
            if old_link is not None:
                nxt = self._next_normal_route_edge(vehicle_id, vehicle_api)
                if nxt is not None and self.topology.edge_to_link.get(nxt) != old_link:
                    self._clear_assignment(vehicle_id)
            return
        physical_link = self.topology.edge_to_link.get(road)
        if physical_link is None:
            if self.current_link.get(vehicle_id) is not None:
                self._clear_assignment(vehicle_id)
            return
        entered_new_link = self.current_link.get(vehicle_id) != physical_link
        if entered_new_link:
            self._clear_assignment(vehicle_id)
            self.current_link[vehicle_id] = physical_link
            self.link_entry_time[vehicle_id] = int(now)
        elif vehicle_id not in self.link_entry_time:
            self.link_entry_time[vehicle_id] = int(now)
        if not entered_new_link and vehicle_id in self.current_movement:
            return
        movement_id, is_bypass = self._movement_on_fixed_link(vehicle_id, physical_link, road, vehicle_api)
        if movement_id is None:
            if next_tls is None:
                try:
                    next_tls = vehicle_api.getNextTLS(vehicle_id)
                except Exception:
                    next_tls = ()
            movement_id = self._movement_from_next_tls_for_link(next_tls, physical_link)
            is_bypass = False
        if movement_id is None:
            self.current_movement.pop(vehicle_id, None)
            self.current_is_bypass.pop(vehicle_id, None)
            self.mapping_failures_total += 1
            return
        self.current_movement[vehicle_id] = movement_id
        self.current_is_bypass[vehicle_id] = bool(is_bypass)

    def add_departed(self, vehicle_ids: Sequence[str], now_s: int, vehicle_api) -> list[str]:
        added=[]
        for vehicle_id in vehicle_ids:
            self.active_connected.add(vehicle_id)
            added.append(vehicle_id)
            try:
                road=str(vehicle_api.getRoadID(vehicle_id))
            except Exception:
                road=""
            if road:
                self._update_vehicle(vehicle_id, road, now_s, vehicle_api)
        return added

    def remove_arrived(self, vehicle_ids: Sequence[str]) -> None:
        for vehicle_id in vehicle_ids:
            self.active_connected.discard(vehicle_id)
            self._clear_assignment(vehicle_id)
            self.last_road.pop(vehicle_id, None)

    def update_roads(self, now_s: int, road_by_vehicle: Mapping[str,str], vehicle_api) -> None:
        for vehicle_id in tuple(self.active_connected):
            road=str(road_by_vehicle.get(vehicle_id, ""))
            if road:
                self._update_vehicle(vehicle_id, road, now_s, vehicle_api)

    def build_state(self, now_s: int):
        raw_w={m:0.0 for m in self.topology.routing_movements}
        by: dict[str,list[VehicleSnapshot]]=defaultdict(list)
        unknown=0
        for vehicle_id in tuple(self.active_connected):
            link_id=self.current_link.get(vehicle_id)
            movement_id=self.current_movement.get(vehicle_id)
            entry=self.link_entry_time.get(vehicle_id)
            if link_id is None or entry is None:
                continue
            if movement_id is None:
                unknown += 1
                continue
            movement_def=self.topology.routing_movements.get(movement_id)
            if movement_def is None or movement_def.source_link != link_id:
                unknown += 1
                continue
            ff=float(self.topology.links[link_id].freeflow_time_s)
            tau=max(float(now_s-entry)/max(ff,1e-9),0.0)
            snap=VehicleSnapshot(vehicle_id,link_id,movement_id,entry,tau,bool(self.current_is_bypass.get(vehicle_id,False)))
            raw_w[movement_id]+=tau
            by[movement_id].append(snap)
        pressure_w={m:0.0 for m in self.topology.movements}
        for movement_id in self.topology.routing_movements:
            if movement_id not in pressure_w:
                pressure_w[movement_id]=0.0
        for movement_id,value in raw_w.items():
            if movement_id in self.topology.movements:
                pressure_w[self.topology.pressure_movement(movement_id)] += float(value)
            else:
                pressure_w[movement_id] += float(value)
        self.last_raw_w=raw_w
        return pressure_w, dict(by), unknown


def next_movement_from_fixed_route(topology: CVRouteTopology, current_movement_id: str, route: Sequence[str], route_index: int) -> str | None:
    current=topology.movements[current_movement_id]
    source_link=topology.links[current.source_link]
    start=max(int(route_index),0)
    try:
        terminal_idx=next(i for i in range(start,len(route)) if route[i]==source_link.terminal_edge)
    except StopIteration:
        return None
    if terminal_idx+1>=len(route) or route[terminal_idx+1]!=current.to_edge:
        return None
    allowed_links=set(current.next_link_candidates)
    for j in range(terminal_idx+1,len(route)-1):
        target=topology.routing_movement_by_edge_pair.get((route[j],route[j+1]))
        if target is None:
            continue
        target_link=topology.routing_movements[target].source_link
        return target if (not allowed_links or target_link in allowed_links) else None
    return None

class FixedTurningEstimator:
    """Rolling turning-intention estimator from passive CVs following their fixed input routes."""
    def __init__(self, topology: CVRouteTopology, window_s: int) -> None:
        self.topology=topology
        self.window_s=int(window_s)
        self.history={m:deque() for m in topology.movements}
        self.last_sampled_movement: dict[str,tuple[str,int]]={}
        self._fallback=self._build_fallback()

    def _build_fallback(self):
        result={}
        for source,targets in self.topology.routing_targets_by_source.items():
            if not targets:
                result[source]={}
                continue
            raw={}
            for target in targets:
                controlled=self.topology.routing_movements[target].controlled_movement_id
                if controlled is not None:
                    raw[target]=float(max(self.topology.movements[controlled].phase_connection_weight.values(),default=1))
                else:
                    raw[target]=1.0
            total=sum(raw.values())
            result[source]={k:v/total for k,v in raw.items()}
        return result

    def observe(self, now_s: int, snapshots: Sequence[VehicleSnapshot], vehicle_api) -> None:
        for snap in snapshots:
            if snap.is_uncontrolled_bypass or snap.movement_id not in self.topology.movements:
                continue
            key=(snap.movement_id,snap.link_entry_time_s)
            if self.last_sampled_movement.get(snap.vehicle_id)==key:
                continue
            try:
                route=tuple(str(e) for e in vehicle_api.getRoute(snap.vehicle_id))
                route_index=int(vehicle_api.getRouteIndex(snap.vehicle_id))
            except Exception:
                continue
            target=next_movement_from_fixed_route(self.topology,snap.movement_id,route,route_index)
            if target is None:
                continue
            self.history[snap.movement_id].append((int(now_s),target))
            self.last_sampled_movement[snap.vehicle_id]=key
        cutoff=int(now_s)-self.window_s
        for q in self.history.values():
            while q and q[0][0] <= cutoff:
                q.popleft()

    def row(self, source_movement: str) -> dict[str,float]:
        q=self.history[source_movement]
        if not q:
            return dict(self._fallback[source_movement])
        counts=defaultdict(float)
        for _,target in q:
            counts[target]+=1.0
        total=sum(counts.values())
        return {k:v/total for k,v in counts.items()} if total>0 else dict(self._fallback[source_movement])

class ConnectedServiceFailureMonitor:
    def __init__(self, topology: CVRouteTopology, config: CVMPConfig) -> None:
        self.topology=topology; self.config=config
        self.history={m:deque(maxlen=config.service_failure_required_intervals) for m in topology.movements}
        self.current_served:set[str]=set(); self.current_waiting={}; self.current_passed=defaultdict(int)
        self.movement_cache={}; self.current_zero_service:set[str]=set(); self.trigger_count=0
        self._served_by_tls_phase={}
        for tls_id,states in topology.phase_states.items():
            reps=[m for m,movement in topology.movements.items() if topology.pressure_movement(m)==m and movement.tls_id==tls_id]
            for phase_id in range(len(states)):
                self._served_by_tls_phase[(tls_id,phase_id)]={m for m in reps if phase_id in topology.movements[m].served_phase_ids}

    def movements_served_by_phase(self,tls_id,phase_id):
        return set(self._served_by_tls_phase.get((tls_id,int(phase_id)),()))

    def initialize_interval(self, selected_phase_by_tls: Mapping[str,int], waiting_connected: Mapping[str,int]|None=None, zero_service:set[str]|None=None):
        served=set()
        for tls_id,phase_id in selected_phase_by_tls.items(): served.update(self.movements_served_by_phase(tls_id,int(phase_id)))
        self.current_served=served; self.current_waiting=dict(waiting_connected or {}); self.current_passed=defaultdict(int); self.current_zero_service=set(zero_service or ())

    def cache_movements(self,snapshots:Sequence[VehicleSnapshot]):
        for snap in snapshots:
            if not snap.is_uncontrolled_bypass:
                self.movement_cache[snap.vehicle_id]=self.topology.pressure_movement(snap.movement_id)

    def observe_arrivals_before_removal(self,tracker:CVMPTracker,arrived_vehicle_ids:Sequence[str]):
        for vehicle_id in arrived_vehicle_ids:
            if vehicle_id not in tracker.active_connected: continue
            old_link=tracker.current_link.get(vehicle_id); old_road=tracker.last_road.get(vehicle_id,"")
            actual=tracker.current_movement.get(vehicle_id)
            movement_id=self.topology.pressure_movement(actual) if actual is not None else self.movement_cache.get(vehicle_id)
            if old_link is None or not old_road or movement_id is None: continue
            link=self.topology.links[old_link]; movement=self.topology.movements.get(movement_id)
            if old_road==link.terminal_edge and movement is not None and movement.source_link==old_link:
                self.current_passed[movement_id]+=1

    def remove_vehicles(self,vehicle_ids:Sequence[str]):
        for v in vehicle_ids: self.movement_cache.pop(v,None)

    def _movement_after_crossing(self,vehicle_id:str,old_link:str,vehicle_api):
        try:
            route=tuple(str(e) for e in vehicle_api.getRoute(vehicle_id)); route_index=int(vehicle_api.getRouteIndex(vehicle_id))
        except Exception: return None
        terminal=self.topology.links[old_link].terminal_edge
        if not route: return None
        upper=min(max(route_index,0),len(route)-1); terminal_idx=None
        for idx in range(upper,-1,-1):
            if route[idx]==terminal: terminal_idx=idx; break
        if terminal_idx is None:
            for idx in range(upper+1,len(route)):
                if route[idx]==terminal: terminal_idx=idx; break
        if terminal_idx is None or terminal_idx+1>=len(route): return None
        return self.topology.movement_by_link_to_edge.get((old_link,route[terminal_idx+1]))

    def observe_road_transitions(self,tracker:CVMPTracker,road_by_vehicle:Mapping[str,str],vehicle_api):
        for vehicle_id in tuple(tracker.active_connected):
            old_link=tracker.current_link.get(vehicle_id); old_road=tracker.last_road.get(vehicle_id,""); new_road=str(road_by_vehicle.get(vehicle_id,""))
            if old_link is None or not old_road or not new_road or new_road==old_road: continue
            if old_road != self.topology.links[old_link].terminal_edge: continue
            actual=tracker.current_movement.get(vehicle_id)
            if actual is None or actual not in self.topology.movements or self.topology.movements[actual].source_link != old_link:
                actual=(None if getattr(tracker, 'detector_only', False) else
                        self._movement_after_crossing(vehicle_id,old_link,vehicle_api))
            if actual is not None:
                self.current_passed[self.topology.pressure_movement(actual)] += 1
            self.movement_cache.pop(vehicle_id,None)

    def finalize_interval_and_triggers(self):
        if not self.config.service_failure_feedback_enabled:
            self.current_passed=defaultdict(int); return set(),[]
        required=self.config.service_failure_required_intervals; min_waiting=self.config.service_failure_min_waiting_connected
        triggers=[]; zero_service=set()
        for movement_id in self.topology.movements:
            if self.topology.pressure_movement(movement_id) != movement_id: continue
            obs=ServiceIntervalObservation(movement_id in self.current_served,int(self.current_waiting.get(movement_id,0)),int(self.current_passed.get(movement_id,0)))
            hist=self.history[movement_id]; hist.append(obs)
            if len(hist)<required: continue
            recent=list(hist)[-required:]
            if all(x.served and x.waiting_connected>=min_waiting and x.passed_connected==0 for x in recent):
                zero_service.add(movement_id)
                triggers.append(ServiceFailureTrigger(movement_id,(recent[-2].waiting_connected,recent[-1].waiting_connected),(recent[-2].passed_connected,recent[-1].passed_connected)))
                hist.clear()
        self.current_zero_service=zero_service; self.trigger_count+=len(triggers)
        return zero_service,triggers

class CVMPController:
    """Pure Route-MP CV-MP signal controller. No path choice is present."""
    def __init__(self, topology: CVRouteTopology, config: CVMPConfig, turning: FixedTurningEstimator) -> None:
        self.topology=topology; self.config=config; self.turning=turning
        self._receiving_slots = {}
        self._receiving_capacity = {}
        self._lane_head_cache = {}
        self._phase_active_lanes = {}
        self._tls_representatives={}; self._phase_representatives={}
        for tls_id,states in topology.phase_states.items():
            reps=tuple(m for m,movement in topology.movements.items() if movement.tls_id==tls_id and topology.pressure_movement(m)==m)
            self._tls_representatives[tls_id]=reps
            for phase_id in range(len(states)):
                self._phase_representatives[(tls_id,phase_id)]=tuple(m for m in reps if phase_id in topology.movements[m].served_phase_ids)
        self._phase_physical_lane_weight={}
        for tls_id,states in topology.phase_states.items():
            for phase_id,state in enumerate(states):
                reps=self._phase_representatives.get((tls_id,phase_id),())
                lanes_by_rep={}; lane_users=defaultdict(set)
                for rep in reps:
                    movement=topology.movements[rep]; lanes=set()
                    for link_index in movement.link_indices:
                        if link_index>=len(state) or state[link_index] not in "gG": continue
                        lane_id=topology.tls_link_from_lane.get((tls_id,int(link_index)))
                        if lane_id is None: raise ValueError(f"missing source lane for {tls_id} linkIndex {link_index}")
                        lanes.add(str(lane_id))
                    lanes_by_rep[rep]=lanes
                    for lane_id in lanes: lane_users[lane_id].add(rep)
                for rep,lanes in lanes_by_rep.items():
                    self._phase_active_lanes[(tls_id,phase_id,rep)] = tuple(sorted(lanes))
                    self._phase_physical_lane_weight[(tls_id,phase_id,rep)] = float(sum(1.0/len(lane_users[l]) for l in lanes))

    def pressure_state(self,w,movement_id):
        if movement_id is None: return 0.0
        key=self.topology.pressure_movement(movement_id) if movement_id in self.topology.movements else movement_id
        return float(w.get(key,0.0))

    def _group_members(self,rep): return self.topology.pressure_group_members.get(rep,(rep,))

    def _group_fixed_row(self,rep,raw_w):
        members=self._group_members(rep); weights={m:max(float(raw_w.get(m,0.0)),0.0) for m in members}; total=sum(weights.values())
        if total<=1e-12: weights={m:1.0 for m in members}; total=float(len(members))
        counts=defaultdict(float)
        for member in members:
            coeff=weights[member]/total
            for target,prob in self.turning.row(member).items(): counts[target]+=coeff*float(prob)
        s=sum(counts.values()); return {k:v/s for k,v in counts.items()} if s>0 else {}

    def begin_decision(self):
        self._receiving_slots.clear()
        self._lane_head_cache.clear()

    def _service_lane_allowed(self, lane):
        return True

    def _lane_head_movement(self, lane):
        if lane not in self._lane_head_cache:
            movement_id = None
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            if vehicles:
                head = max(vehicles, key=traci.vehicle.getLanePosition)
                edge = lane.rsplit("_", 1)[0]
                route = traci.vehicle.getRoute(head)
                index = CVMPTracker._route_position(
                    route, traci.vehicle.getRouteIndex(head), edge)
                if index is not None and index + 1 < len(route):
                    movement_id = self.topology.movement_by_edge_pair.get(
                        (edge, route[index + 1]))
            self._lane_head_cache[lane] = movement_id
        return self._lane_head_cache[lane]

    def _serviceable_lane_weight(self, movement, phase_id):
        tls = movement.tls_id
        state = self.topology.phase_states[tls][int(phase_id)]
        lanes = self._phase_active_lanes.get((tls, int(phase_id), movement.movement_id), ())
        weight = 0.0
        for lane in lanes:
            if not self._service_lane_allowed(lane):
                continue
            actual_id = self._lane_head_movement(lane)
            if actual_id is None or self.topology.pressure_movement(actual_id) != movement.movement_id:
                continue
            actual = self.topology.movements[actual_id]
            if any(index < len(state) and state[index] in "gG"
                   and self.topology.tls_link_from_lane[(tls, index)] == lane
                   for index in actual.link_indices):
                # A shared lane supplies its full capacity only to its head's movement.
                weight += 1.0
        return weight

    def service_amount(self,movement:MovementDef,phase_id:int,service_factor:float):
        weight=self._serviceable_lane_weight(movement,phase_id)
        if weight<=0: return 0.0
        nominal = self.config.saturation_flow_vph_per_lane/3600.0*weight*self.config.control_interval_s*service_factor
        edge = movement.to_edge
        if edge not in self._receiving_slots:
            if edge not in self._receiving_capacity:
                lanes = traci.edge.getLaneNumber(edge)
                lengths = [traci.lane.getLength(f'{edge}_{i}') for i in range(lanes)]
                self._receiving_capacity[edge] = max(1.0, sum(lengths) / 7.5)
            capacity = self._receiving_capacity[edge]
            count = traci.edge.getLastStepVehicleNumber(edge)
            occupancy = traci.edge.getLastStepOccupancy(edge) / 100.0
            self._receiving_slots[edge] = max(0.0, capacity - max(count, capacity * occupancy))
        return min(nominal, self._receiving_slots[edge])

    def service_capacity_audit(self):
        rows=[]
        for (tls_id,phase_id),reps in sorted(self._phase_representatives.items()):
            state=self.topology.phase_states[tls_id][phase_id]
            for rep in reps:
                m=self.topology.movements[rep]
                active=sorted({self.topology.tls_link_from_lane[(tls_id,int(i))] for i in m.link_indices if i<len(state) and state[i] in "gG"})
                rows.append({"tls_id":tls_id,"phase_id":phase_id,"movement_id":rep,"corrected_physical_lane_weight":self._phase_physical_lane_weight.get((tls_id,phase_id,rep),0.0),"active_source_lanes":" ".join(active)})
        return rows

    def evaluate_phase(self,tls_id,phase_id,current_phase,w,raw_w,temporarily_zero_service=frozenset()):
        service_factor=1.0 if phase_id==current_phase else self.config.switch_service_factor
        total=0.0; details={}
        for movement_id in self._phase_representatives.get((tls_id,phase_id),()):
            m=self.topology.movements[movement_id]
            mu_nom=self.service_amount(m,phase_id,service_factor); zeroed=movement_id in temporarily_zero_service; mu=0.0 if zeroed else mu_nom
            rf=self._group_fixed_row(movement_id,raw_w)
            downstream=sum(float(prob)*self.pressure_state(w,target) for target,prob in rf.items())
            diff=float(w.get(movement_id,0.0))-downstream
            raw_local=mu*diff; local=mu*max(diff,0.0); total += local
            details[movement_id]={"mu_veh":mu,"mu_nominal_veh":mu_nom,"service_temporarily_zero":float(zeroed),"w_up":float(w.get(movement_id,0.0)),"downstream_w":downstream,"pressure_difference_raw":diff,"raw_pressure":raw_local,"pressure":local}
        return PhaseEvaluation(phase_id,total,total,service_factor,details)

    def select_phase(self,tls_id,current_phase,w,raw_w,temporarily_zero_service=frozenset()):
        evaluations=[self.evaluate_phase(tls_id,p,current_phase,w,raw_w,temporarily_zero_service) for p in range(len(self.topology.phase_states[tls_id]))]
        best=max(evaluations,key=lambda e:(e.score,int(e.phase_id==current_phase),-e.phase_id))
        return best,evaluations

class SimpleSignalCoordinator:
    def __init__(self,phase_states:Sequence[str],yellow_s:int,interval_s:int,maximum_red_s:int=180,start_time_s:int=0):
        self.phase_states=tuple(str(s) for s in phase_states); self.yellow_s=int(yellow_s); self.interval_s=int(interval_s); self.maximum_red_s=int(maximum_red_s)
        if not self.phase_states: raise ValueError("TLS has no legal green phase")
        self.current_phase=0; self.last_served_s=[int(start_time_s)]*len(self.phase_states); self.last_reason="initial"; self.last_requested_phase=0
    def round_robin_next_phase(self): return (self.current_phase+1)%len(self.phase_states)
    def _deadline_order(self): return tuple(sorted((p for p in range(len(self.phase_states)) if p!=self.current_phase),key=lambda p:(self.last_served_s[p]+self.maximum_red_s,p)))
    def _would_violate_max_red(self,target,now_s):
        ordered=self._deadline_order()
        if not ordered: return False
        now=int(now_s)
        if target==self.current_phase: proposed=ordered; first_start=now+self.interval_s+self.yellow_s
        else: proposed=(target,)+tuple(p for p in ordered if p!=target); first_start=now+self.yellow_s
        return any(first_start+i*self.interval_s > self.last_served_s[p]+self.maximum_red_s for i,p in enumerate(proposed))
    def enforce_max_red(self,requested_phase,now_s):
        target=int(requested_phase); self.last_requested_phase=target; self.last_reason="selected"
        if self._would_violate_max_red(target,int(now_s)):
            ordered=self._deadline_order()
            if ordered: target=int(ordered[0]); self.last_reason="maximum_red"
        return target
    def schedule(self,selected_phase,now_s):
        now=int(now_s); self.last_served_s[self.current_phase]=now
        if selected_phase==self.current_phase: return [self.phase_states[self.current_phase]]*self.interval_s
        old=self.phase_states[self.current_phase]; new=self.phase_states[selected_phase]
        yellow="".join("y" if a in "Gg" and b not in "Gg" else a for a,b in zip(old,new))
        out=[yellow]*self.yellow_s+[new]*(self.interval_s-self.yellow_s)
        self.current_phase=int(selected_phase); self.last_served_s[self.current_phase]=now+self.yellow_s
        return out
