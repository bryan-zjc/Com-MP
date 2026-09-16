# -*- coding: utf-8 -*-
"""
Summarize SUMO tripinfo outputs for each method in a tested network.

Default example:
    python tripinfo_statistics.py

Other examples:
    python tripinfo_statistics.py --intersection 5x5
    python tripinfo_statistics.py --all
"""

import argparse
import csv
import math
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_INTERSECTION = "qinzhou"
DEFAULT_TRIPINFO_ROOT = "tripinfo"
OUTPUT_FILE_NAME = "tripinfo_efficiency_summary.csv"
DEFAULT_WARM_START_STEPS = 150


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _percentile(values, q):
    if not values:
        return 0.0

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[int(pos)]

    weight = pos - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def _method_name(xml_path):
    stem = xml_path.stem
    if stem.lower().startswith("tripinfo_"):
        return stem[len("tripinfo_"):]
    return stem


def _parse_penetration_rate(xml_path):
    match = re.search(r"cvpr_([0-9]+(?:p[0-9]+)?)", xml_path.stem, re.IGNORECASE)
    if not match:
        return ""
    return _to_float(match.group(1).replace("p", "."))


def _round_row(row, digits=4):
    rounded = {}
    for key, value in row.items():
        if isinstance(value, float):
            rounded[key] = round(value, digits)
        else:
            rounded[key] = value
    return rounded


def _iter_tripinfo_elements(xml_path, warm_start_steps=DEFAULT_WARM_START_STEPS):
    with xml_path.open("r", encoding="utf-8", errors="ignore") as xml_file:
        for line in xml_file:
            line = line.strip()
            if not line.startswith("<tripinfo "):
                continue
            try:
                elem = ET.fromstring(line)
            except ET.ParseError:
                continue

            if warm_start_steps > 0:
                depart = _to_float(elem.attrib.get("depart"))
                depart_delay = _to_float(elem.attrib.get("departDelay"))
                intended_depart = depart - depart_delay
                if intended_depart < warm_start_steps:
                    continue

            yield elem


def summarize_tripinfo_file(xml_path, warm_start_steps=DEFAULT_WARM_START_STEPS):
    durations = []
    route_lengths = []
    vehicle_speeds_mps = []
    arrival_speeds = []
    waiting_times = []
    waiting_counts = []
    stop_times = []
    time_losses = []
    total_delays = []
    depart_delays = []
    arrivals = []
    vtype_counts = {}

    for elem in _iter_tripinfo_elements(xml_path, warm_start_steps):
        duration = _to_float(elem.attrib.get("duration"))
        route_length = _to_float(elem.attrib.get("routeLength"))
        waiting_time = _to_float(elem.attrib.get("waitingTime"))
        waiting_count = _to_int(elem.attrib.get("waitingCount"))
        stop_time = _to_float(elem.attrib.get("stopTime"))
        time_loss = _to_float(elem.attrib.get("timeLoss"))
        depart_delay = _to_float(elem.attrib.get("departDelay"))
        total_delay = time_loss + depart_delay
        arrival = _to_float(elem.attrib.get("arrival"))
        arrival_speed = _to_float(elem.attrib.get("arrivalSpeed"))
        vtype = elem.attrib.get("vType", "unknown")

        durations.append(duration)
        route_lengths.append(route_length)
        waiting_times.append(waiting_time)
        waiting_counts.append(waiting_count)
        stop_times.append(stop_time)
        time_losses.append(time_loss)
        total_delays.append(total_delay)
        depart_delays.append(depart_delay)
        arrivals.append(arrival)
        arrival_speeds.append(arrival_speed)
        vtype_counts[vtype] = vtype_counts.get(vtype, 0) + 1

        if duration > 0:
            vehicle_speed_mps = route_length / duration
            vehicle_speeds_mps.append(vehicle_speed_mps)

    vehicle_count = len(durations)
    total_route_length_km = sum(route_lengths) / 1000.0
    total_duration_h = sum(durations) / 3600.0
    evaluation_duration_s = max(arrivals) - warm_start_steps if arrivals else 0.0
    throughput_veh_per_h = vehicle_count / (evaluation_duration_s / 3600.0) if evaluation_duration_s > 0 else 0.0

    row = {
        "method": _method_name(xml_path),
        "file_name": xml_path.name,
        "v_penetration_rate": _parse_penetration_rate(xml_path),
        "warm_start_steps": warm_start_steps,
        "evaluation_duration_s": evaluation_duration_s,
        "vehicle_count": vehicle_count,
        "cv_count": vtype_counts.get("CV", 0),
        "nv_count": vtype_counts.get("NV", 0),
        "total_route_length_km": total_route_length_km,
        "total_travel_time_h": total_duration_h,
        "throughput_veh_per_h": throughput_veh_per_h,
        "delay_definition": "timeLoss_plus_departDelay",
        "avg_delay_s": _mean(total_delays),
        "median_delay_s": statistics.median(total_delays) if total_delays else 0.0,
        "p95_delay_s": _percentile(total_delays, 0.95),
        "max_delay_s": max(total_delays) if total_delays else 0.0,
        "total_delay_h": sum(total_delays) / 3600.0,
        "avg_time_loss_s": _mean(time_losses),
        "total_time_loss_h": sum(time_losses) / 3600.0,
        "avg_travel_time_s": _mean(durations),
        "median_travel_time_s": statistics.median(durations) if durations else 0.0,
        "p95_travel_time_s": _percentile(durations, 0.95),
        "max_travel_time_s": max(durations) if durations else 0.0,
        "avg_speed_mps": _mean(vehicle_speeds_mps),
        "avg_speed_kmh": _mean(vehicle_speeds_mps) * 3.6,
        "avg_arrival_speed_mps": _mean(arrival_speeds),
        "avg_arrival_speed_kmh": _mean(arrival_speeds) * 3.6,
        "avg_waiting_time_s": _mean(waiting_times),
        "p95_waiting_time_s": _percentile(waiting_times, 0.95),
        "max_waiting_time_s": max(waiting_times) if waiting_times else 0.0,
        "total_waiting_time_h": sum(waiting_times) / 3600.0,
        "avg_stop_count": _mean(waiting_counts),
        "total_stop_count": sum(waiting_counts),
        "max_stop_count": max(waiting_counts) if waiting_counts else 0,
        "avg_stop_time_s": _mean(stop_times),
        "total_stop_time_h": sum(stop_times) / 3600.0,
        "avg_depart_delay_s": _mean(depart_delays),
        "p95_depart_delay_s": _percentile(depart_delays, 0.95),
        "last_arrival_s": max(arrivals) if arrivals else 0.0,
    }
    return _round_row(row)


def summarize_intersection(intersection, tripinfo_root=DEFAULT_TRIPINFO_ROOT,
                           warm_start_steps=DEFAULT_WARM_START_STEPS):
    tripinfo_dir = Path(tripinfo_root) / intersection
    if not tripinfo_dir.is_dir():
        raise FileNotFoundError(f"Cannot find tripinfo directory: {tripinfo_dir}")

    xml_files = sorted(
        path for path in tripinfo_dir.glob("*.xml")
        if path.name.lower().startswith("tripinfo_")
    )
    if not xml_files:
        raise FileNotFoundError(f"No tripinfo_*.xml files found in: {tripinfo_dir}")

    rows = [summarize_tripinfo_file(xml_path, warm_start_steps) for xml_path in xml_files]
    output_path = tripinfo_dir / OUTPUT_FILE_NAME

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path, rows


def summarize_all(tripinfo_root=DEFAULT_TRIPINFO_ROOT,
                  warm_start_steps=DEFAULT_WARM_START_STEPS):
    root = Path(tripinfo_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Cannot find tripinfo root: {root}")

    outputs = []
    for tripinfo_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        output_path, rows = summarize_intersection(tripinfo_dir.name,
                                                   tripinfo_root,
                                                   warm_start_steps)
        outputs.append((output_path, rows))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Summarize SUMO tripinfo efficiency metrics.")
    parser.add_argument("--intersection", default=DEFAULT_INTERSECTION,
                        help="Network folder name under tripinfo/. Default: qinzhou")
    parser.add_argument("--tripinfo-root", default=DEFAULT_TRIPINFO_ROOT,
                        help="Root folder containing tripinfo outputs. Default: tripinfo")
    parser.add_argument("--warm-start-steps", type=int, default=DEFAULT_WARM_START_STEPS,
                        help="Ignore vehicles departing before this time. Default: 150")
    parser.add_argument("--all", action="store_true",
                        help="Summarize every network folder under tripinfo-root.")
    args = parser.parse_args()

    if args.all:
        outputs = summarize_all(args.tripinfo_root, args.warm_start_steps)
        for output_path, rows in outputs:
            print(f"Saved {len(rows)} method rows to {output_path}")
    else:
        output_path, rows = summarize_intersection(args.intersection,
                                                   args.tripinfo_root,
                                                   args.warm_start_steps)
        print(f"Saved {len(rows)} method rows to {output_path}")


if __name__ == "__main__":
    main()
