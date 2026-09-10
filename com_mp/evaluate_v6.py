import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import traci

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_v6 import ExperimentConfig
from env_hybrid_mappo_v6 import DetectorSlimHybridMAPPOEnv
from src.utils import utils
from tripinfo_statistics import summarize_tripinfo_file


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def make_env(config, episode_seed, tripinfo_path):
    tls_ids = utils.extract_trafficlight_ids(str(config.net_file))
    sumo_cmd = utils.set_sumo(
        config.gui, str(config.sumocfg_file), config.max_steps)
    sumo_cmd += ["--seed", str(episode_seed)]
    env = DetectorSlimHybridMAPPOEnv(
        str(config.rou_file), str(config.net_file), len(tls_ids),
        sumo_cmd, config.max_steps, episode_seed,
        cv_penetration_rate=config.cv_penetration_rate,
        alpha_bins=config.alpha_bins,
        queue_reduction_reward_weight=(
            config.queue_reduction_reward_weight),
        queue_level_reward_weight=config.queue_level_reward_weight,
        queue_change_clip=config.queue_change_clip,
        reward_scale=config.reward_scale,
        max_red_seconds=config.max_red_seconds,
        gamma_per_second=config.gamma_per_second,
        decision_interval_seconds=config.decision_interval_seconds,
    )
    observations, _, decision_flags = env.reset(str(tripinfo_path))
    if config.warm_start_steps:
        _, decision_flags = env.fixed_time_warm_start(
            config.warm_start_steps)
    env.begin_control()
    observations = env.get_all_observations()
    return env, tls_ids, observations, decision_flags


def run_policy(config, method, seed, checkpoint=None, output_dir=None):
    output = Path(output_dir) if output_dir else config.output_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    tripinfo = output / f"tripinfo_{method}_seed{seed}.xml"
    spillover_path = output / f"spillover_{method}_seed{seed}.csv"
    time_series_path = output / f"timeseries_{method}_seed{seed}.csv"
    decisions_path = output / f"decisions_{method}_seed{seed}.csv"
    env, tls_ids, observations, decision_flags = make_env(config, seed, tripinfo)
    if method != "HybridMAPPO":
        # All methods share service feasibility, yellow loss and maximum red.
        env.controller_mode = method
        env.track_waiting_state = False
        if method == "MP":
            env.track_cv_state = False
        elif method == "CVMP":
            env._decision_interval = 10
            env.hold_current_on_cvmp_tie = True
            env.cv_absence_fixed_time_fallback = True
    env.reward_active = False
    agent = None
    if method == "HybridMAPPO":
        import torch
        from hybrid_mappo_agent import AsyncMAPPOAgent

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_size = len(observations[tls_ids[0]])
        agent = AsyncMAPPOAgent(obs_size, config.alpha_bins, config, device)
        agent.load(checkpoint)
        agent.network.eval()
    alpha_values = []
    decision_history = []
    time_series_history = []
    spillover_summary = {}
    spillover_history = []

    def record_network_state():
        queued = sum(
            float(env._lane_value(
                lane_id, traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER))
            for lane_id in env.lane_capacity
        )
        spillover = (
            float(env.spillover_count_history[-1][1])
            if env.spillover_count_history else
            float(len(traci.simulation.getPendingVehicles()))
        )
        time_series_history.append({
            "simulation_time_s": float(traci.simulation.getTime()),
            "network_vehicle_count": int(traci.vehicle.getIDCount()),
            "controlled_approach_queue_count": queued,
            "source_spillover_count": spillover,
        })

    try:
        record_network_state()
        done = False
        while not done:
            actions = {}
            if any(decision_flags.get(t, 0) for t in tls_ids):
                if method == "HybridMAPPO":
                    obs_matrix = env.get_observation_matrix(tls_ids)
                    context = env.get_global_context(obs_matrix)
                for idx, tls_id in enumerate(tls_ids):
                    if not decision_flags.get(tls_id, 0):
                        continue
                    diagnostics = env.get_action_diagnostics(tls_id)
                    if method == "MP":
                        action_index = 0
                        alpha = 0.0
                    elif method == "CVMP":
                        action_index = len(config.alpha_bins) - 1
                        alpha = 1.0
                    elif method == "FixedAlpha075":
                        action_index = min(
                            range(len(config.alpha_bins)),
                            key=lambda item: abs(
                                config.alpha_bins[item] - 0.75))
                        alpha = 0.75
                    else:
                        action_index, alpha, _, _ = agent.act(
                            obs_matrix[idx], context,
                            env.get_action_mask(tls_id),
                            diagnostics["phase_by_action"],
                            deterministic=True)
                    actions[tls_id] = alpha
                    alpha_values.append(alpha)
                    decision_history.append({
                        "simulation_time_s": float(
                            traci.simulation.getTime()),
                        "tls_id": tls_id,
                        "action_index": int(action_index),
                        "alpha": float(alpha),
                        "selected_phase": int(
                            diagnostics["phase_by_action"][action_index]),
                        "mp_phase": int(diagnostics["mp_phase"]),
                        "cvmp_phase": int(diagnostics["cvmp_phase"]),
                        "actor_valid": int(diagnostics["actor_valid"]),
                        "candidate_phase_count": int(
                            diagnostics["candidate_phase_count"]),
                    })
            _, dones, decision_flags, _ = env.step(actions)
            record_network_state()
            done = any(dones[t][0] for t in tls_ids)
    finally:
        spillover_summary = env.get_source_spillover_summary()
        spillover_history = list(env.spillover_count_history)
        env.close()
    with spillover_path.open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("simulation_time_s", "pending_insertion_count"))
        writer.writerows(spillover_history)
    with time_series_path.open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(time_series_history[0]))
        writer.writeheader()
        writer.writerows(time_series_history)
    with decisions_path.open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(decision_history[0]))
        writer.writeheader()
        writer.writerows(decision_history)
    summary = summarize_tripinfo_file(tripinfo, config.warm_start_steps)
    return {**summary, **spillover_summary, "method": method, "seed": seed,
            "avg_alpha": float(np.mean(alpha_values)) if alpha_values else 0.0,
            "v_penetration_rate": config.cv_penetration_rate}


def evaluate(config, checkpoint, seeds, methods=None):
    set_seed(config.seed)
    rows = []
    methods = methods or ("MP", "CVMP", "FixedAlpha075", "HybridMAPPO")
    for seed in seeds:
        for method in methods:
            row = run_policy(config, method, seed, checkpoint)
            rows.append(row)
            print(f"[EVAL] {method} seed={seed} delay={row['avg_delay_s']:.3f}s "
                  f"speed={row['avg_speed_kmh']:.3f}km/h "
                  f"spillover_max={row['max_source_spillover_count']} "
                  f"alpha={row['avg_alpha']:.3f}",
                  flush=True)
    path = config.output_dir / "evaluation_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    aggregate(rows, config.output_dir / "evaluation_aggregate.csv")


def aggregate(rows, path):
    metrics = ("avg_delay_s", "avg_speed_kmh", "avg_waiting_time_s",
               "avg_stop_count", "vehicle_count",
               "max_source_spillover_count",
               "mean_source_spillover_count",
               "final_source_spillover_count",
               "spillover_vehicle_seconds",
               "spillover_last_300s_slope_veh_per_s")
    output = []
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        item = {"method": method, "seeds": len(selected)}
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
        output.append(item)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(THIS_DIR / "checkpoints" / "5x5" /
                    "com_mp_v0909" / "best.pt"))
    parser.add_argument("--seeds", default="20260801,20260802,20260803")
    parser.add_argument("--intersection", default="5x5")
    parser.add_argument(
        "--experiment-name", default="com_mp_v0909")
    parser.add_argument("--cv-rate", type=float, default=0.5)
    parser.add_argument("--methods", default="MP,CVMP,FixedAlpha075,HybridMAPPO")
    args = parser.parse_args()
    cfg = ExperimentConfig(intersection=args.intersection,
                           experiment_name=args.experiment_name,
                           cv_penetration_rate=args.cv_rate)
    evaluate(cfg, Path(args.checkpoint), [int(x) for x in args.seeds.split(",")],
             [x.strip() for x in args.methods.split(",") if x.strip()])
