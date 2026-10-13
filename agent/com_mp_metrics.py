"""Completed-trip metrics for the fixed-horizon Com-MP training runs."""

import xml.etree.ElementTree as ET


def summarize_completed_trips(path):
    delays = []
    speeds = []
    waiting = []
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "tripinfo":
            continue
        arrival = float(element.get("arrival", "-1"))
        depart = float(element.get("depart", "-1"))
        if arrival >= 0 and depart >= 0:
            duration = float(element.get("duration", "0"))
            route_length = float(element.get("routeLength", "0"))
            delays.append(float(element.get("timeLoss", "0"))
                          + float(element.get("departDelay", "0")))
            waiting.append(float(element.get("waitingTime", "0")))
            if duration > 0:
                speeds.append(route_length / duration * 3.6)
        element.clear()

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "avg_delay_s": mean(delays),
        "avg_speed_kmh": mean(speeds),
        "avg_waiting_time_s": mean(waiting),
        "vehicle_count": len(delays),
    }
