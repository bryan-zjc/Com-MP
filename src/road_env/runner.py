from __future__ import annotations
from collections import defaultdict, deque
from datetime import datetime
import csv, hashlib, json, math, time, xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping

from .sumo_bootstrap import ensure_sumo_python_tools
ensure_sumo_python_tools()
import sumolib
import traci
import traci.constants as tc

from .config import CVMPConfig, default_paths
from .env_CVMP import CVMPTracker, FixedTurningEstimator, ConnectedServiceFailureMonitor, CVMPController, SimpleSignalCoordinator
from .topology import load_topology

ROOT=Path(__file__).resolve().parents[2]

class CsvStream:
    def __init__(self,path:Path,fieldnames):
        path.parent.mkdir(parents=True,exist_ok=True); self.f=path.open('w',newline='',encoding='utf-8-sig'); self.w=csv.DictWriter(self.f,fieldnames=fieldnames); self.w.writeheader()
    def write(self,row): self.w.writerow(row)
    def close(self): self.f.close()

class ProgressBar:
    def __init__(self,total,label): self.total=max(int(total),1); self.label=label; self.started=time.monotonic(); self.last=-1
    def update(self,now,force=False,status=''):
        if not force and int(now)==self.last: return
        self.last=int(now); frac=min(max(now/self.total,0),1); width=28; filled=int(round(width*frac)); elapsed=time.monotonic()-self.started; eta=elapsed*(1/frac-1) if frac>0 else float('nan')
        def fmt(x):
            if not math.isfinite(x): return '--:--:--'
            x=int(round(x)); h,r=divmod(x,3600); m,s=divmod(r,60); return f'{h:02d}:{m:02d}:{s:02d}'
        print(f"\r{self.label:<16} [{'#'*filled}{'-'*(width-filled)}] {100*frac:5.1f}% sim={now:5d}/{self.total}s elapsed={fmt(elapsed)} ETA={fmt(eta)} {status}",end='\n' if force else '',flush=True)


def _expected_demand(route_file:Path)->float:
    root=ET.parse(route_file).getroot(); total=float(len(root.findall('vehicle')) + len(root.findall('trip')))
    for flow in root.findall('flow'):
        p=str(flow.attrib.get('period','')).strip(); rate=float(p[4:-1]) if p.startswith('exp(') and p.endswith(')') else 0.0
        total += rate*(float(flow.attrib.get('end','0'))-float(flow.attrib.get('begin','0')))
    return total

def _is_connected(vehicle_id:str,seed:int,p:float)->bool:
    raw=int.from_bytes(hashlib.sha256(f'LYG-CVMP|{seed}|{vehicle_id}'.encode()).digest()[:8],'big'); u=raw/float(1<<64); return u<float(p)

def _rgb(text):
    a=[int(x.strip()) for x in text.split(',')]; return a[0],a[1],a[2],255

def _sumo_command(cfg,paths,out):
    args=[str(sumolib.checkBinary('sumo-gui' if cfg.gui else 'sumo'))]
    if cfg.gui:
        args += ['--start','--quit-on-end','true']
        if paths.get('gui_settings') is not None:
            args += ['--gui-settings-file',str(paths['gui_settings'].resolve())]
    args += ['--net-file',str(paths['network'].resolve()),'--route-files',str(paths['route'].resolve()),'--begin','0','--end',str(cfg.maximum_simulation_s),'--step-length','1','--seed',str(cfg.sumo_seed),'--time-to-teleport','300','--ignore-junction-blocker','-1','--collision.action','warn','--no-step-log','true','--log',str((out/'sumo.log').resolve())]
    if cfg.save_tripinfo: args += ['--tripinfo-output',str((out/'tripinfo.xml').resolve()),'--tripinfo-output.write-unfinished','true','--tripinfo-output.write-undeparted','true']
    return args

def _subscribe_edges(edge_ids):
    for edge in edge_ids:
        try: traci.edge.subscribe(edge,[tc.LAST_STEP_HALTING_NUMBER])
        except Exception: pass

def _halting():
    total=0
    for vals in (traci.edge.getAllSubscriptionResults() or {}).values():
        try: total += int(vals.get(tc.LAST_STEP_HALTING_NUMBER,0) or 0)
        except Exception: pass
    return total

def _subscribe_connected(ids):
    for v in ids:
        try: traci.vehicle.subscribe(v,[tc.VAR_ROAD_ID,tc.VAR_SPEED,tc.VAR_LANEPOSITION])
        except Exception: pass

def _subscription_frame(active):
    allr=traci.vehicle.getAllSubscriptionResults() or {}; roads={}; speeds={}
    for v in active:
        vals=allr.get(v,{}) or {}
        try: roads[v]=str(vals[tc.VAR_ROAD_ID] if tc.VAR_ROAD_ID in vals else traci.vehicle.getRoadID(v))
        except Exception: pass
        try: speeds[v]=float(vals[tc.VAR_SPEED] if tc.VAR_SPEED in vals else traci.vehicle.getSpeed(v))
        except Exception: pass
    return roads,speeds

def _aggregate_counts(topology,counts):
    out=defaultdict(int)
    for mid,n in counts.items():
        key=topology.pressure_movement(mid) if mid in topology.movements else mid; out[key]+=int(n)
    return dict(out)

def _trip_summary(path:Path,residual:int):
    if not path.is_file(): return {'tripinfo_available':False,'residual_vehicles':residual}
    root=ET.parse(path).getroot(); trips=list(root.findall('tripinfo'))
    def f(e,k,d=-1):
        try:return float(e.attrib.get(k,d))
        except:return float(d)
    inserted=[t for t in trips if f(t,'depart')>=0]; completed=[t for t in inserted if f(t,'arrival')>=0]
    def vals(k,rows): return [f(t,k,0) for t in rows if f(t,k,0)>=0]
    dur=vals('duration',completed); tl=vals('timeLoss',inserted); wait=vals('waitingTime',inserted); dd=vals('departDelay',trips)
    return {'tripinfo_available':True,'tripinfo_records':len(trips),'completed_trips':len(completed),'inserted_trips':len(inserted),'residual_vehicles':residual,'mean_duration_s':sum(dur)/len(dur) if dur else None,'mean_time_loss_s':sum(tl)/len(tl) if tl else None,'mean_waiting_time_s':sum(wait)/len(wait) if wait else None,'total_time_loss_s':sum(tl)+sum(dd),'total_time_loss_h':(sum(tl)+sum(dd))/3600.0}

def run(cfg:CVMPConfig,output:Path|None=None,*,paths=None,connected_selector=None):
    cfg.validate(); paths=default_paths(ROOT) if paths is None else paths
    connected_selector = _is_connected if connected_selector is None else connected_selector
    for k in ('network','topology','route'):
        if not paths[k].is_file(): raise FileNotFoundError(paths[k])
    topology,graph=load_topology(paths['topology'],paths['network'])
    out=Path(output) if output else ROOT/'results'/f"cvmp_s{cfg.sumo_seed:03d}_cv{cfg.connected_penetration:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}".replace('.','p')
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg.to_json_dict(),indent=2)+'\n',encoding='utf-8')
    decisions=CsvStream(out/'control_decisions.csv',['time_s','tls_id','control_mode','connected_at_tls','current_phase','selected_phase','switched','pressure_score','score_margin_to_second','temporarily_zero_service_movements'])
    networklog=CsvStream(out/'network_metrics.csv',['time_s','active_vehicles','active_connected','total_halting_vehicles','arrived_cumulative','phase_switches_cumulative','zero_pressure_round_robin_decisions_cumulative','maximum_red_overrides_cumulative','movement_mapping_failures_total','control_epoch_wall_s'])
    servicelog=CsvStream(out/'service_feedback.csv',['time_s','movement_id','tls_id','action','waiting_prev2_interval1','waiting_prev2_interval2','passed_prev2_interval1','passed_prev2_interval2'])
    cmd=_sumo_command(cfg,paths,out)
    traci.start(cmd,numRetries=3)
    _subscribe_edges(graph.edge_length_m)
    tracker=CVMPTracker(topology,graph); turning=FixedTurningEstimator(topology,cfg.turning_window_s); controller=CVMPController(topology,cfg,turning); monitor=ConnectedServiceFailureMonitor(topology,cfg)
    with (out/'service_capacity_audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
        rows=controller.service_capacity_audit(); w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    coords={tls:SimpleSignalCoordinator(states,cfg.yellow_s,cfg.control_interval_s,cfg.maximum_red_s) for tls,states in topology.phase_states.items()}
    schedules={tls:deque([states[0]]*cfg.control_interval_s) for tls,states in topology.phase_states.items()}; last_state={}
    for tls,states in topology.phase_states.items(): traci.trafficlight.setRedYellowGreenState(tls,states[0]); last_state[tls]=states[0]
    monitor.initialize_interval({tls:c.current_phase for tls,c in coords.items()})
    bar=ProgressBar(cfg.maximum_simulation_s,'CV-MP')
    now=0; arrived_total=0; departed_total=0; connected_departed=0; phase_switches=0; rr=0; maxred=0; service_triggers=0; halting_sum=0; halting_n=0; peak_halt=0; residual=0; latest_speeds={}; teleports=0
    wall0=time.monotonic()
    try:
        while now<cfg.maximum_simulation_s:
            if now>0 and now%cfg.control_interval_s==0:
                tick=time.monotonic(); zero_service,triggers=monitor.finalize_interval_and_triggers(); service_triggers += len(triggers)
                for x in triggers:
                    m=topology.movements[x.movement_id]; servicelog.write({'time_s':now,'movement_id':x.movement_id,'tls_id':m.tls_id,'action':'ZERO_SERVICE_NEXT_INTERVAL','waiting_prev2_interval1':x.waiting_prev2[0],'waiting_prev2_interval2':x.waiting_prev2[1],'passed_prev2_interval1':x.passed_prev2[0],'passed_prev2_interval2':x.passed_prev2[1]})
                controller.begin_decision(); state,by,unknown=tracker.build_state(now); raw=tracker.last_raw_w; snaps=[s for xs in by.values() for s in xs]; monitor.cache_movements(snaps); turning.observe(now,snaps,traci.vehicle)
                waiting_raw={mid:sum(1 for s in ss if latest_speeds.get(s.vehicle_id,float('inf'))<=cfg.service_failure_waiting_speed_mps) for mid,ss in by.items()}; waiting=_aggregate_counts(topology,waiting_raw)
                connected_tls={tls:0 for tls in coords}
                for mid,ss in by.items():
                    m=topology.movements.get(mid)
                    if m is not None: connected_tls[m.tls_id]+=len(ss)
                chosen={}
                for tls,c in coords.items():
                    cur=c.current_phase; best,evals=controller.select_phase(tls,cur,state,raw,zero_service); mode='cv_pressure'
                    if all(ev.pressure_score<=0.0 for ev in evals):
                        pid=c.round_robin_next_phase(); best=next(ev for ev in evals if ev.phase_id==pid); mode='round_robin_zero_pressure'; rr+=1
                    req=best.phase_id; forced=c.enforce_max_red(req,now)
                    if forced!=req: best=next(ev for ev in evals if ev.phase_id==forced); mode='maximum_red'; maxred+=1
                    switched=int(best.phase_id!=cur); phase_switches += switched; chosen[tls]=best; schedules[tls]=deque(c.schedule(best.phase_id,now))
                    scores=sorted((float(ev.pressure_score) for ev in evals),reverse=True); margin=scores[0]-scores[1] if len(scores)>1 else scores[0]
                    decisions.write({'time_s':now,'tls_id':tls,'control_mode':mode,'connected_at_tls':connected_tls[tls],'current_phase':cur,'selected_phase':best.phase_id,'switched':switched,'pressure_score':best.pressure_score,'score_margin_to_second':margin,'temporarily_zero_service_movements':' '.join(sorted(zero_service))})
                monitor.initialize_interval({tls:b.phase_id for tls,b in chosen.items()},waiting_connected=waiting,zero_service=zero_service)
                h=_halting(); halting_sum+=h; halting_n+=1; peak_halt=max(peak_halt,h)
                networklog.write({'time_s':now,'active_vehicles':traci.vehicle.getIDCount(),'active_connected':len(tracker.active_connected),'total_halting_vehicles':h,'arrived_cumulative':arrived_total,'phase_switches_cumulative':phase_switches,'zero_pressure_round_robin_decisions_cumulative':rr,'maximum_red_overrides_cumulative':maxred,'movement_mapping_failures_total':tracker.mapping_failures_total,'control_epoch_wall_s':time.monotonic()-tick})
                if cfg.show_progress: bar.update(now,status=f'act={traci.vehicle.getIDCount()} halt={h} arr={arrived_total}')
            for tls,schedule in schedules.items():
                if not schedule: raise RuntimeError(f'empty signal schedule: {tls} t={now}')
                s=schedule[0]
                if last_state.get(tls)!=s: traci.trafficlight.setRedYellowGreenState(tls,s); last_state[tls]=s
            traci.simulationStep(); now+=1
            for s in schedules.values():
                if s:s.popleft()
            try: teleports += len(traci.simulation.getStartingTeleportIDList())
            except Exception: pass
            departed=[str(v) for v in traci.simulation.getDepartedIDList()]; arrived=[str(v) for v in traci.simulation.getArrivedIDList()]; arrived_set=set(arrived)
            monitor.observe_arrivals_before_removal(tracker,arrived); arrived_total+=len(arrived); tracker.remove_arrived(arrived); monitor.remove_vehicles(arrived)
            live=[v for v in departed if v not in arrived_set]; departed_total+=len(live); new_connected=[]
            for v in live:
                if connected_selector(v,cfg.sumo_seed,cfg.connected_penetration):
                    new_connected.append(v); connected_departed+=1
                    try: traci.vehicle.setColor(v,_rgb(cfg.cv_color))
                    except Exception: pass
                else:
                    try: traci.vehicle.setColor(v,_rgb(cfg.nv_color))
                    except Exception: pass
            tracker.add_departed(new_connected,now,traci.vehicle); _subscribe_connected(new_connected)
            roads,latest_speeds=_subscription_frame(tracker.active_connected); monitor.observe_road_transitions(tracker,roads,traci.vehicle); tracker.update_roads(now,roads,traci.vehicle)
            if now>=cfg.demand_end_s and traci.simulation.getMinExpectedNumber()<=0: break
        residual=int(traci.vehicle.getIDCount())
    finally:
        try: traci.close(True)
        finally:
            decisions.close(); networklog.close(); servicelog.close();
            if cfg.show_progress: bar.update(now,force=True)
    summary=_trip_summary(out/'tripinfo.xml',residual); summary.update({'schema':'lyg-cvmp-route-mp-direct-stripped/v2','method':'cv_mp','sumo_seed':cfg.sumo_seed,'connected_penetration':cfg.connected_penetration,'simulation_end_s':now,'expected_demand_vehicle_count':_expected_demand(paths['route']),'realized_departed_vehicle_count':departed_total,'realized_connected_departed':connected_departed,'realized_connected_share':connected_departed/departed_total if departed_total else None,'arrived_cumulative':arrived_total,'completion_rate':summary.get('completed_trips',arrived_total)/summary.get('tripinfo_records',1) if summary.get('tripinfo_records',0) else None,'phase_switches':phase_switches,'zero_pressure_round_robin_decisions':rr,'maximum_red_overrides':maxred,'service_failure_triggers':service_triggers,'mean_total_halting_vehicles':halting_sum/halting_n if halting_n else None,'peak_halting_vehicles':peak_halt,'teleport_events':teleports,'movement_mapping_failures_total':tracker.mapping_failures_total,'wall_clock_runtime_s':time.monotonic()-wall0,'route_updates':0,'routing_enabled':False})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('\nCV-MP finished:',out); print('completed=',summary.get('completed_trips'),' mean delay=',summary.get('mean_time_loss_s'),' route_updates=0')
    return summary
