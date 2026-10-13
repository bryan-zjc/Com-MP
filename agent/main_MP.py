"""Run topology-aware Q-MP on Qinzhou."""
from datetime import datetime
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.road_env.build_topology import build_topology
from src.road_env.config import CVMPConfig
from src.road_env.runner_MP import run as run_qmp
from src.road_env.topology import load_topology


# Edit these values to run directly in Spyder.
INTERSECTION = "qinzhou"
SCENARIO = "main"  # main, high_cv, or static
SUMO_SEED = 20260801
GUI = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intersection", choices=("qinzhou",), default=INTERSECTION)
    parser.add_argument("--scenario", choices=("main", "high_cv", "static"), default=SCENARIO)
    parser.add_argument("--seed", type=int, default=SUMO_SEED)
    parser.add_argument("--route", type=Path, help="Existing SUMO route file.")
    parser.add_argument("--max-seconds", type=int, help="Default: 3600 grid / 15000 dynamic cap.")
    parser.add_argument("--gui", action="store_true", default=GUI)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def scenario_paths(args):
    network = ROOT / "envs" / args.intersection / f"{args.intersection}.net.xml"
    scenario = args.scenario
    dynamic = args.intersection == "qinzhou" and scenario in ("main", "high_cv")
    if dynamic:
        folder = ("qinzhou_dynamic_3h_moderate" if scenario == "main"
                  else "qinzhou_dynamic_3h_11k_15k_13k_cv04_07_05")
        route = ROOT / "scenarios" / folder / f"Demand_dynamic_seed{args.seed}.rou.xml"
    elif scenario == "static":
        route = ROOT / "envs" / args.intersection / "Demand.rou.xml"
    else:
        raise ValueError(f"Unsupported network/scenario: {args.intersection}/{scenario}")
    if args.route is not None:
        route = args.route.resolve()
    for path in (network, route):
        if not path.is_file():
            raise FileNotFoundError(path)
    return network, route, scenario, dynamic


def main():
    args = parse_args()
    network, route, scenario, dynamic = scenario_paths(args)
    demand_end = 10800 if dynamic else 1800
    max_seconds = args.max_seconds if args.max_seconds is not None else (15000 if dynamic else demand_end)
    if max_seconds <= 0:
        raise ValueError("max-seconds must be positive")
    out = args.output or (ROOT / "results" / args.intersection / "q-mp" /
                          scenario / f"seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}")
    topology_path = ROOT / "envs" / args.intersection / "topology.json"
    if topology_path.is_file():
        load_topology(topology_path, network)
    else:
        build_topology(network, topology_path)
    cfg = CVMPConfig(
        sumo_seed=args.seed, connected_penetration=0.0,
        control_interval_s=15, yellow_s=3, maximum_red_s=180,
        turning_window_s=300, maximum_simulation_s=max_seconds,
        demand_end_s=demand_end, gui=args.gui, show_progress=False,
    )
    result = run_qmp(cfg, output=out, paths={"network": network, "topology": topology_path,
                                "route": route})
    print(f"Q-MP finished: {out}")
    print(result)


if __name__ == "__main__":
    main()
