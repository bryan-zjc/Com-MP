import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

if int(np.__version__.split(".", 1)[0]) >= 2:
    raise RuntimeError(
        "This training environment requires NumPy < 2 because the installed "
        "PyTorch/SciPy stack was compiled against the NumPy 1.x ABI. "
        "Install numpy==1.26.4 and restart the Python/Spyder kernel.")

import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_INTERSECTION = "qinzhou"
SUPPORTED_INTERSECTIONS = ("qinzhou",)
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

from com_mp_config import ExperimentConfig
from dynamic_training_demand import ensure_dynamic_training_route
from com_mp_env import DetectorSlimHybridMAPPOEnv
from com_mp_mappo import AsyncMAPPOAgent
from com_mp_metrics import summarize_completed_trips
from src.road_env.build_topology import build_topology
from src.road_env.topology import load_topology


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_network_files(config):
    if config.intersection not in SUPPORTED_INTERSECTIONS:
        supported = ", ".join(SUPPORTED_INTERSECTIONS)
        raise ValueError(
            f"Unsupported intersection '{config.intersection}'. "
            f"Choose one of: {supported}.")
    required = (config.net_file, config.rou_file)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing SUMO network file(s):\n" + "\n".join(missing))


def entropy_schedule(config, episode):
    if config.episodes <= 1:
        return config.entropy_final
    fraction = episode / (config.episodes - 1)
    return config.entropy_initial * ((config.entropy_final /
                                      config.entropy_initial) ** fraction)


def make_env(config, episode_seed, tripinfo_path):
    topology_file = ROOT / "envs" / config.intersection / "topology.json"
    topology = (load_topology(topology_file, config.net_file)[0]
                if topology_file.is_file() else
                build_topology(config.net_file, topology_file)[0])
    tls_ids = sorted(topology.phase_states)
    if config.dynamic_training_scenario:
        if config.intersection != "qinzhou":
            raise ValueError(
                "The dynamic 3 h training profile is defined for Qinzhou.")
        route_file = ensure_dynamic_training_route(
            config.rou_file, config.dynamic_training_dir, episode_seed)
    else:
        route_file = config.rou_file
    env = DetectorSlimHybridMAPPOEnv(
        str(route_file), str(config.net_file), config.max_steps, episode_seed,
        cv_penetration_rate=config.cv_penetration_rate,
        alpha_bins=config.alpha_bins,
        queue_reduction_reward_weight=config.queue_reduction_reward_weight,
        queue_level_reward_weight=config.queue_level_reward_weight,
        reward_scale=config.reward_scale,
        gamma_per_second=config.gamma_per_second,
        decision_interval_seconds=config.decision_interval_seconds,
        route_defined_cv=config.dynamic_training_scenario,
        stop_exactly_at_max_steps=True,
    )
    observations, _, decision_flags = env.reset(str(tripinfo_path))
    if config.warm_start_steps:
        _, decision_flags = env.fixed_time_warm_start(config.warm_start_steps)
    env.begin_control()
    observations = env.get_all_observations()
    return env, tls_ids, observations, decision_flags


def training_learning_rates(agent, episode, keep_current=False):
    if keep_current:
        return tuple(group["lr"] for group in agent.optimizer.param_groups)
    return agent.set_learning_rate(episode)


def train(config, resume_path=None, keep_resume_learning_rate=False):
    if keep_resume_learning_rate and not resume_path:
        raise ValueError("--resume-keep-learning-rate requires --resume.")
    validate_network_files(config)
    if not config.dynamic_training_scenario:
        raise ValueError("Qinzhou requires dynamic_training_scenario=True.")
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = None
    history = []
    best_delay = float("inf")
    start_episode = 0
    resume_metadata = None
    wandb_run = init_wandb(config, device)

    if resume_path:
        resume_payload = torch.load(resume_path, map_location="cpu",
                                    weights_only=False)
        resume_metadata = resume_payload.get("metadata", {})
        if resume_payload.get("obs_size") != DetectorSlimHybridMAPPOEnv.OBSERVATION_SIZE:
            raise ValueError(
                "This version requires 70-dimensional phase-context observations. "
                "Old 34-dimensional checkpoints cannot be resumed; start a fresh experiment.")
        if keep_resume_learning_rate and "optimizer" not in resume_payload:
            raise ValueError("Checkpoint has no optimizer state to preserve learning rates.")
        if resume_metadata.get("controller_backend") != "v8_movement_queue":
            raise ValueError(
                "The checkpoint does not match the current movement-queue observations. "
                "Start a fresh training run.")
        saved_interval = resume_metadata.get(
            "config", {}).get("decision_interval_seconds")
        if saved_interval != config.decision_interval_seconds:
            raise ValueError(
                "Resume checkpoint decision interval does not match the "
                f"current {config.decision_interval_seconds}s configuration. "
                "Start a fresh training run.")
        start_episode = int(resume_metadata.get("episode", 0))
        best_delay = float(resume_metadata.get("best_validation_delay_s", best_delay))

    for episode in range(start_episode, config.episodes):
        # Keep the SUMO demand realization and CV assignment fixed across episodes.
        episode_seed = config.seed
        tripinfo_path = config.output_dir / f"train_ep{episode + 1:03d}.xml"
        env = None
        pending = {}
        actions_taken = []
        action_counts = np.zeros(len(config.alpha_bins), dtype=np.int64)
        decision_count = 0
        actor_valid_count = 0
        method_conflict_count = 0
        selected_mp_count = 0
        selected_cvmp_count = 0
        candidate_phase_counts = []
        episode_reward = 0.0
        reward_components = {"queue_reduction": 0.0, "queue_level": 0.0}
        try:
            env, tls_ids, observations, decision_flags = make_env(
                config, episode_seed, tripinfo_path)
            obs_matrix = np.stack([observations[t] for t in tls_ids])
            if agent is None:
                agent = AsyncMAPPOAgent(obs_matrix.shape[1], config.alpha_bins,
                                        config, device)
                if resume_path:
                    agent.load(resume_path, load_optimizer=True)
                    if keep_resume_learning_rate:
                        actor_lr, critic_lr = training_learning_rates(agent, episode, True)
                        print(f"Keeping checkpoint learning rates: actor={actor_lr:.6e}, "
                              f"critic={critic_lr:.6e}", flush=True)
            done = False
            while not done:
                alpha_actions = {}
                if any(decision_flags.get(tls_id, 0) for tls_id in tls_ids):
                    obs_matrix = env.get_observation_matrix(tls_ids)
                    global_context = env.get_global_context(obs_matrix)
                    for idx, tls_id in enumerate(tls_ids):
                        if not decision_flags.get(tls_id, 0):
                            continue
                        if tls_id in pending:
                            reward, duration, components = env.consume_interval_reward(tls_id)
                            next_diagnostics = env.get_action_diagnostics(tls_id)
                            _, _, _, next_value = agent.act(
                                obs_matrix[idx], global_context,
                                env.get_action_mask(tls_id),
                                next_diagnostics["phase_by_action"],
                                deterministic=True)
                            transition = pending.pop(tls_id)
                            transition.update(reward=reward, duration=duration,
                                              next_value=next_value, done=False)
                            agent.add_transition(tls_id, transition)
                            episode_reward += reward
                            for key in reward_components:
                                reward_components[key] += components.get(key, 0.0)
                        else:
                            # The first action begins after the initial 15-second interval.
                            env.reset_interval_reward(tls_id)

                        mask = env.get_action_mask(tls_id)
                        diagnostics = env.get_action_diagnostics(tls_id)
                        action, alpha, log_prob, value = agent.act(
                            obs_matrix[idx], global_context, mask,
                            diagnostics["phase_by_action"], deterministic=False)
                        alpha_actions[tls_id] = alpha
                        actions_taken.append(alpha)
                        action_counts[action] += 1
                        decision_count += 1
                        actor_valid_count += int(diagnostics["actor_valid"])
                        method_conflict_count += int(
                            diagnostics["mp_phase"] != diagnostics["cvmp_phase"])
                        selected_phase = diagnostics["phase_by_action"][action]
                        selected_mp_count += int(selected_phase == diagnostics["mp_phase"])
                        selected_cvmp_count += int(
                            selected_phase == diagnostics["cvmp_phase"])
                        candidate_phase_counts.append(
                            diagnostics["candidate_phase_count"])
                        pending[tls_id] = {
                            "obs": obs_matrix[idx].copy(),
                            "context": global_context.copy(),
                            "mask": mask.copy(),
                            "action": action,
                            "log_prob": log_prob,
                            "value": value,
                            "actor_valid": diagnostics["actor_valid"],
                            "phase_by_action": np.asarray(
                                diagnostics["phase_by_action"], dtype=np.int64),
                        }

                _, dones, decision_flags, _ = env.step(alpha_actions)
                done = any(dones[tls_id][0] for tls_id in tls_ids)

            final_obs = env.get_observation_matrix(tls_ids)
            final_context = env.get_global_context(final_obs)
            for idx, tls_id in enumerate(tls_ids):
                if tls_id not in pending:
                    continue
                reward, duration, components = env.consume_interval_reward(tls_id)
                transition = pending[tls_id]
                transition.update(reward=reward, duration=duration,
                                  next_value=0.0, done=True)
                agent.add_transition(tls_id, transition)
                episode_reward += reward
                for key in reward_components:
                    reward_components[key] += components.get(key, 0.0)

            env.close()
            env = None
            actor_lr, critic_lr = training_learning_rates(
                agent, episode, keep_resume_learning_rate)
            entropy_coef = entropy_schedule(config, episode)
            update_log = agent.update(entropy_coef)
            summary = summarize_completed_trips(tripinfo_path)
            validation_delay = None
            validation_speed = None
            if should_validate(config, episode + 1):
                validation = run_validation(config, agent, episode + 1)
                validation_delay = validation["avg_delay_s"]
                validation_speed = validation["avg_speed_kmh"]
            row = {
                "episode": episode + 1,
                "seed": episode_seed,
                "episode_reward": episode_reward,
                "avg_alpha": float(np.mean(actions_taken)) if actions_taken else 0.0,
                "alpha_std": float(np.std(actions_taken)) if actions_taken else 0.0,
                "entropy_coef": entropy_coef,
                "actor_learning_rate": actor_lr,
                "critic_learning_rate": critic_lr,
                "avg_delay_s": summary["avg_delay_s"],
                "avg_speed_kmh": summary["avg_speed_kmh"],
                "avg_waiting_time_s": summary["avg_waiting_time_s"],
                "vehicle_count": summary["vehicle_count"],
                "reward_queue_reduction": reward_components["queue_reduction"],
                "reward_queue_level": reward_components["queue_level"],
                "reward_queue_reduction_contribution": (
                    config.reward_scale * config.queue_reduction_reward_weight *
                    reward_components["queue_reduction"]),
                "reward_queue_level_contribution": (
                    -config.reward_scale * config.queue_level_reward_weight *
                    reward_components["queue_level"]),
                "decision_count": decision_count,
                "actor_valid_fraction": safe_ratio(actor_valid_count, decision_count),
                "method_conflict_fraction": safe_ratio(method_conflict_count,
                                                         decision_count),
                "selected_mp_fraction": safe_ratio(selected_mp_count, decision_count),
                "selected_cvmp_fraction": safe_ratio(selected_cvmp_count,
                                                       decision_count),
                "mean_candidate_phase_count": (
                    float(np.mean(candidate_phase_counts))
                    if candidate_phase_counts else 0.0),
                "validation_avg_delay_s": (validation_delay
                                             if validation_delay is not None else ""),
                "validation_avg_speed_kmh": (validation_speed
                                              if validation_speed is not None else ""),
                **update_log,
            }
            for idx, alpha in enumerate(config.alpha_bins):
                key = f"alpha_{alpha:.2f}_fraction".replace(".", "p")
                row[key] = safe_ratio(int(action_counts[idx]), decision_count)
            history.append(row)
            append_csv(config.output_dir / "training_history.csv", row)
            if validation_delay is not None and validation_delay < best_delay:
                best_delay = validation_delay
            metadata = {"episode": episode + 1,
                        "controller_backend": "v8_movement_queue",
                        "observation_schema": DetectorSlimHybridMAPPOEnv.OBSERVATION_SCHEMA,
                        "config": public_config(config),
                        "metrics": row,
                        "best_validation_delay_s": best_delay}
            agent.save(config.checkpoint_dir / "latest.pt", metadata)
            if validation_delay is not None and validation_delay == best_delay:
                agent.save(config.checkpoint_dir / "best.pt", metadata)
            if wandb_run:
                wandb_run.log(row, step=episode + 1)
            validation_text = (f" val_delay={validation_delay:.3f}s"
                               if validation_delay is not None else "")
            print("[Com-MP-v8] ep={}/{} reward={:.3f} delay={:.3f}s "
                  "speed={:.3f}km/h alpha={:.3f}+/-{:.3f} valid={:.1%} "
                  "entropy={:.3f} ev={:.3f} actor_lr={:.2e} "
                  "critic_lr={:.2e}{}".format(
                      episode + 1, config.episodes, episode_reward,
                      summary["avg_delay_s"], summary["avg_speed_kmh"],
                      row["avg_alpha"], row["alpha_std"],
                      row["actor_valid_fraction"],
                      update_log.get("entropy", 0.0),
                      update_log.get("explained_variance", 0.0),
                      actor_lr,
                      critic_lr,
                      validation_text), flush=True)
        finally:
            if env is not None:
                env.close()

    if wandb_run:
        wandb_run.finish()
    return history


def append_csv(path, row):
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            existing_fields = next(csv.reader(handle), [])
        if existing_fields != fieldnames:
            raise ValueError(
                f"History schema mismatch in {path}. Start a new experiment_name.")
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def init_wandb(config, device):
    if not config.use_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb unavailable; training continues locally.")
        return None
    # Uses WANDB_API_KEY from the environment or an existing `wandb login`.
    wandb.login()
    wandb_dir = ROOT / "outputs" / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=config.wandb_project,
        name=(f"HybridMAPPO-{config.intersection}-"
              f"{config.experiment_name}-cvpr{config.cv_penetration_rate:.2f}-"
              f"seed{config.seed}"),
        dir=str(wandb_dir),
        config={**public_config(config), "device": str(device)},
    )
    run.define_metric("episode")
    run.define_metric("*", step_metric="episode")
    return run


def public_config(config):
    """Return serializable experiment settings."""
    return dict(vars(config))


def safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def should_validate(config, episode):
    return (config.validation_interval > 0 and
            (episode % config.validation_interval == 0 or
             episode == config.episodes))


def run_validation(config, agent, episode):
    """Evaluate deterministic policy on fixed seeds for checkpoint selection."""
    delays = []
    speeds = []
    agent.network.eval()
    try:
        for seed in config.validation_seeds:
            output = config.output_dir / "validation"
            output.mkdir(parents=True, exist_ok=True)
            tripinfo = output / f"ep{episode:03d}_seed{seed}.xml"
            env = None
            try:
                env, tls_ids, _, decision_flags = make_env(
                    config, int(seed), tripinfo)
                env.reward_active = False
                done = False
                while not done:
                    actions = {}
                    if any(decision_flags.get(tls_id, 0) for tls_id in tls_ids):
                        observations = env.get_observation_matrix(tls_ids)
                        context = env.get_global_context(observations)
                        for idx, tls_id in enumerate(tls_ids):
                            if not decision_flags.get(tls_id, 0):
                                continue
                            _, alpha, _, _ = agent.act(
                                observations[idx], context,
                                env.get_action_mask(tls_id),
                                env.get_action_diagnostics(tls_id)["phase_by_action"],
                                deterministic=True)
                            actions[tls_id] = alpha
                    _, dones, decision_flags, _ = env.step(actions)
                    done = any(dones[tls_id][0] for tls_id in tls_ids)
                env.close()
                env = None
                summary = summarize_completed_trips(tripinfo)
                delays.append(float(summary["avg_delay_s"]))
                speeds.append(float(summary["avg_speed_kmh"]))
            finally:
                if env is not None:
                    env.close()
    finally:
        agent.network.train()
    return {"avg_delay_s": float(np.mean(delays)),
            "avg_speed_kmh": float(np.mean(speeds))}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intersection",
        choices=SUPPORTED_INTERSECTIONS,
        default=TRAIN_INTERSECTION,
        help="Training network. Defaults to TRAIN_INTERSECTION in this file.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Episode horizon. Defaults to 10800 s for Qinzhou.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cv-rate", type=float, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--validation-interval", type=int, default=20)
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument("--wandb", dest="use_wandb", action="store_true")
    wandb_group.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.set_defaults(use_wandb=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-keep-learning-rate", action="store_true",
                        help="Keep checkpoint optimizer learning rates fixed throughout resumed training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    is_qinzhou = True
    experiment_name = args.experiment_name or "com_mp_v9_phase_context_dynamic_3h"
    max_steps = args.max_steps or 10800
    seed = args.seed if args.seed is not None else 1
    cv_rate = args.cv_rate if args.cv_rate is not None else 0.5
    if not 0.0 <= cv_rate <= 1.0:
        raise ValueError("--cv-rate must be between 0 and 1.")
    cfg = ExperimentConfig(experiment_name=experiment_name,
                           intersection=args.intersection,
                           episodes=args.episodes,
                           max_steps=max_steps,
                           seed=seed,
                           cv_penetration_rate=cv_rate,
                           validation_interval=args.validation_interval,
                           use_wandb=args.use_wandb,
                           dynamic_training_scenario=is_qinzhou)
    validate_network_files(cfg)
    print(f"Training network: {cfg.intersection}")
    print(f"Network: {cfg.net_file}")
    demand_profile = cfg.dynamic_training_profile
    print(f"Training demand profile: {demand_profile}")
    print(f"Episode horizon: {cfg.max_steps}s (no clearance extension)")
    print(f"Decision interval: {cfg.decision_interval_seconds}s")
    print(f"Fixed training seed for every episode: {cfg.seed}")
    train(cfg, Path(args.resume) if args.resume else None,
          keep_resume_learning_rate=args.resume_keep_learning_rate)
