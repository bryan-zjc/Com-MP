"""Convert the Qinzhou direct-link network to the topology schema used by v9."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .topology import PhysicalEdgeGraph, load_topology


def build_topology(network: Path, output: Path):
    root = ET.parse(network).getroot()
    graph = PhysicalEdgeGraph(network)
    phases = {}
    for logic in root.findall("tlLogic"):
        tls = logic.attrib["id"]
        if tls in phases:
            raise ValueError(f"Multiple signal programs require explicit selection: {tls}")
        phases[tls] = [p.attrib["state"] for p in logic.findall("phase")
                       if "y" not in p.attrib["state"].lower()
                       and any(c in "gG" for c in p.attrib["state"])]
        if not phases[tls]:
            raise ValueError(f"No green phases: {tls}")

    grouped = defaultdict(list)
    source_tls = {}
    for conn in root.findall("connection"):
        a = conn.attrib
        if "tl" not in a:
            continue
        tls, edge, target = a["tl"], a["from"], a["to"]
        if edge not in graph.edge_length_m or target not in graph.edge_length_m:
            continue
        if source_tls.setdefault(edge, tls) != tls:
            raise ValueError(f"Approach belongs to multiple TLS: {edge}")
        grouped[(tls, edge, target)].append(a)

    edge_links = {edge: f"L:{tls}:{edge}" for edge, tls in source_tls.items()}
    links = {}
    for edge, tls in source_tls.items():
        upstream = graph.edge_from_junction[edge]
        if upstream not in phases and graph.reverse[edge]:
            raise ValueError(f"Non-direct approach requires a logical-link adapter: {edge}")
        links[edge_links[edge]] = {
            "tls_id": tls, "downstream_tls": tls,
            "upstream_tls": upstream if upstream in phases else "",
            "boundary_upstream": int(upstream not in phases),
            "terminal_edge": edge, "start_edge": edge, "edges": [edge],
            "length_m": graph.edge_length_m[edge], "movement_ids": [],
        }
    movements = {}
    for (tls, edge, target), connections in grouped.items():
        source = edge_links[edge]
        mid = f"M:{source}->{target}"
        indices = sorted({int(c["linkIndex"]) for c in connections})
        weights = {p: sum(state[i] in "gG" for i in indices)
                   for p, state in enumerate(phases[tls])}
        served = [p for p, count in weights.items() if count]
        if not served:
            raise ValueError(f"Controlled movement has no green phase: {mid}")
        if target not in edge_links and graph.adjacency[target]:
            raise ValueError(f"Non-direct downstream link requires an adapter: {target}")
        movements[mid] = {
            "source_link": source, "tls_id": tls, "to_edge": target,
            "directions": sorted({c.get("dir", "s") for c in connections}),
            "link_indices": indices, "served_phase_ids": served,
            "phase_connection_weight": {p: weights[p] for p in served},
            "next_link_candidates": [edge_links[target]] if target in edge_links else [],
        }
        links[source]["movement_ids"].append(mid)
    payload = {
        "schema": "lyg-clean-full-link-topology/v1",
        "network_sha256": hashlib.sha256(network.read_bytes()).hexdigest(),
        "adapter": "direct-approach-edges/v1",
        "links": links, "movements": movements, "phase_states": phases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return load_topology(output, network)
