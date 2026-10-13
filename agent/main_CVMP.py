"""Run CV-MP on the Qinzhou experiment inputs."""
from pathlib import Path
import argparse
from datetime import datetime
import json
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.road_env.config import CVMPConfig, default_paths
from src.road_env.build_topology import build_topology
from src.road_env.topology import load_topology
from src.road_env.runner import run

# Edit these settings when running this file directly in Spyder.
INTERSECTION = "qinzhou"
SCENARIO = "main"  # main, high_cv, or static
SUMO_SEED = 20260801
CV_PENETRATION = 0.80  # Used only when route files do not contain saved CV types.
CV_SOURCE = "auto"  # Qinzhou dynamic scenarios use saved route types.
GUI = False
MAXIMUM_RED_SECONDS = 180
DECISION_INTERVAL_SECONDS = 15
YELLOW_SECONDS = 3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intersection", choices=("qinzhou",), default=INTERSECTION)
    parser.add_argument("--scenario", choices=("main", "high_cv", "static"), default=SCENARIO)
    parser.add_argument("--seed", type=int, default=SUMO_SEED)
    parser.add_argument("--cv-penetration", type=float, default=CV_PENETRATION)
    parser.add_argument("--cv-source", choices=("auto", "route_type"), default=CV_SOURCE)
    parser.add_argument("--route", type=Path, help="Optional existing demand file; never regenerated.")
    parser.add_argument("--max-seconds", type=int, help="Default: 3600 grid / 15000 dynamic safety cap.")
    parser.add_argument("--gui", action="store_true", default=GUI)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true", help="Build/check topology without SUMO.")
    return parser.parse_args()


def prepare(args):
    paths = default_paths(ROOT, args.intersection)
    scenario = args.scenario
    dynamic = args.intersection == "qinzhou" and scenario in ("main", "high_cv")
    if dynamic:
        folder = ("qinzhou_dynamic_3h_moderate" if scenario == "main"
                  else "qinzhou_dynamic_3h_11k_15k_13k_cv04_07_05")
        paths["route"] = ROOT / "scenarios" / folder / f"Demand_dynamic_seed{args.seed}.rou.xml"
    elif scenario != "static":
        raise ValueError(f"Unsupported network/scenario: {args.intersection}/{scenario}")
    if args.route is not None:
        paths["route"] = args.route.resolve()
    for key in ("network", "route"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    topology = (load_topology(paths["topology"], paths["network"])[0]
                if paths["topology"].is_file() else
                build_topology(paths["network"], paths["topology"])[0])
    cv_source = args.cv_source
    if cv_source == "auto":
        cv_source = "route_type" if dynamic else "hash"
    selector = None
    if cv_source == "route_type":
        vehicles = ET.parse(paths["route"]).getroot().findall("vehicle")
        if not vehicles or any(v.get("type") not in ("CV", "NV") for v in vehicles):
            raise ValueError("route_type requires explicit vehicles typed CV or NV.")
        connected = {v.attrib["id"] for v in vehicles if v.get("type") == "CV"}
        selector = lambda vehicle_id, seed, p: vehicle_id in connected
    demand_end = 10800 if dynamic else 1800
    cfg = CVMPConfig(
        sumo_seed=args.seed, connected_penetration=args.cv_penetration,
        control_interval_s=DECISION_INTERVAL_SECONDS, yellow_s=YELLOW_SECONDS,
        maximum_red_s=MAXIMUM_RED_SECONDS,
        maximum_simulation_s=args.max_seconds if args.max_seconds is not None else (15000 if dynamic else demand_end),
        demand_end_s=demand_end, gui=args.gui, show_progress=not args.quiet,
    )
    cfg.validate()
    return cfg, paths, topology, selector, cv_source, scenario


def main():
    args = parse_args()
    cfg, paths, topology, selector, cv_source, scenario = prepare(args)
    print(f"CV-MP v4: {args.intersection}/{scenario}; TLS={len(topology.phase_states)}; "
          f"interval={cfg.control_interval_s}s; max red={cfg.maximum_red_s}s; CV source={cv_source}")
    print(f"Demand: {paths['route']}")
    if args.validate_only:
        return
    out = args.output or (ROOT / "results" / args.intersection / "cv-mp" / scenario /
                          f"seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"intersection": args.intersection, "scenario": scenario,
                "cv_source": cv_source, "paths": {k: str(v) for k, v in paths.items()},
                "note": "route_type preserves the saved Qinzhou scenario CV identities."}
    (out / "scenario.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    run(cfg, output=out, paths=paths, connected_selector=selector)


if __name__ == "__main__":
    main()
