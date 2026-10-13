from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


DYNAMIC_TRAINING_STAGES = (
    {
        "stage": "low_1",
        "start_s": 0,
        "end_s": 3600,
        "flow_veh_per_h": 11000,
        "cv_penetration_rate": 0.20,
    },
    {
        "stage": "high",
        "start_s": 3600,
        "end_s": 7200,
        "flow_veh_per_h": 15000,
        "cv_penetration_rate": 0.40,
    },
    {
        "stage": "low_2",
        "start_s": 7200,
        "end_s": 10800,
        "flow_veh_per_h": 13000,
        "cv_penetration_rate": 0.30,
    },
)


def _read_route_pool(path):
    routes = []
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag.endswith("vehicle"):
            route = next(
                (child for child in element
                 if child.tag.endswith("route")), None)
            if route is not None and route.attrib.get("edges"):
                routes.append(route.attrib["edges"])
            element.clear()
    if not routes:
        raise ValueError(f"No vehicle routes found in {path}")
    return routes


def _stage_vehicle_count(stage):
    duration_hours = (stage["end_s"] - stage["start_s"]) / 3600.0
    return int(round(stage["flow_veh_per_h"] * duration_hours))


def ensure_dynamic_training_route(source_route_file, output_dir, seed):
    """Create the fixed-seed 3 h Qinzhou training demand when absent."""
    source_route_file = Path(source_route_file)
    output_dir = Path(output_dir)
    output_path = output_dir / f"Demand_dynamic_seed{int(seed)}.rou.xml"
    if output_path.is_file():
        return output_path

    route_pool = _read_route_pool(source_route_file)
    frequencies = Counter(route_pool)
    unique_routes = sorted(frequencies)
    route_weights = np.asarray(
        [frequencies[route] for route in unique_routes], dtype=float)
    route_weights /= route_weights.sum()
    rng = np.random.default_rng(int(seed))

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        handle.write(
            '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:noNamespaceSchemaLocation="'
            'http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        common = (
            'length="5.00" minGap="2.00" maxSpeed="60.00" '
            'speedFactor="normc(1.00,0.00)" guiShape="passenger" '
            'accel="3" decel="5" sigma="0.5" tau="1.0"'
        )
        handle.write(f'    <vType id="NV" {common} color="green"/>\n')
        handle.write(f'    <vType id="CV" {common} color="blue"/>\n')
        for route_index, edges in enumerate(unique_routes):
            handle.write(
                f'    <route id="dynamic_route_{route_index}" '
                f'edges="{edges}"/>\n')

        for stage_index, stage in enumerate(DYNAMIC_TRAINING_STAGES):
            count = _stage_vehicle_count(stage)
            start = float(stage["start_s"])
            end = float(stage["end_s"])
            departures = np.sort(rng.uniform(start, end, size=count))
            departures = np.minimum(departures, np.nextafter(end, start))
            route_indices = rng.choice(
                len(unique_routes), size=count, p=route_weights)
            cv_count = int(round(count * stage["cv_penetration_rate"]))
            cv_mask = np.zeros(count, dtype=bool)
            cv_mask[:cv_count] = True
            rng.shuffle(cv_mask)

            for vehicle_index, (depart, route_index, is_cv) in enumerate(
                    zip(departures, route_indices, cv_mask)):
                vehicle_type = "CV" if is_cv else "NV"
                vehicle_id = (
                    f"dyn_{seed}_{stage_index}_{vehicle_index}_{vehicle_type}")
                handle.write(
                    f'    <vehicle id="{vehicle_id}" type="{vehicle_type}" '
                    f'route="dynamic_route_{int(route_index)}" '
                    f'depart="{depart:.2f}" departSpeed="10"/>\n')
        handle.write("</routes>\n")

    temporary_path.replace(output_path)
    return output_path
