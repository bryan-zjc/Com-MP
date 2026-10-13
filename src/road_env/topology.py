from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Iterable, Mapping
import xml.etree.ElementTree as ET


@dataclass(frozen=True, slots=True)
class LinkDef:
    link_id: str
    tls_id: str
    upstream_tls: str | None
    downstream_tls: str
    boundary_upstream: bool
    terminal_edge: str
    start_edge: str
    edges: tuple[str, ...]
    length_m: float
    movement_ids: tuple[str, ...]
    freeflow_time_s: float


@dataclass(frozen=True, slots=True)
class MovementDef:
    movement_id: str
    source_link: str
    tls_id: str
    to_edge: str
    directions: tuple[str, ...]
    link_indices: tuple[int, ...]
    served_phase_ids: tuple[int, ...]
    phase_connection_weight: Mapping[int, int]
    next_link_candidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoutingMovementDef:
    """Immediate routing movement on one fixed logical link.

    Controlled movements reuse their existing movement ID. Uncontrolled early
    exits (the network's bypass/right-turn movements) receive a synthetic ID.
    Routing movements define the L/T/R action space and historical movement-TT
    state; only controlled movements participate in TLS phase service.
    """
    movement_id: str
    source_link: str
    from_edge: str
    to_edge: str
    direction: str
    controlled_movement_id: str | None
    freeflow_time_s: float


@dataclass(frozen=True, slots=True)
class CVRouteTopology:
    links: Mapping[str, LinkDef]
    movements: Mapping[str, MovementDef]
    phase_states: Mapping[str, tuple[str, ...]]
    movement_by_tls_link_index: Mapping[tuple[str, int], str]
    movement_by_link_to_edge: Mapping[tuple[str, str], str]
    movement_by_edge_pair: Mapping[tuple[str, str], str]
    edge_to_link: Mapping[str, str]
    routing_movements: Mapping[str, RoutingMovementDef]
    routing_movement_by_edge_pair: Mapping[tuple[str, str], str]
    routing_movements_by_source_link: Mapping[str, tuple[str, ...]]
    routing_targets_by_source: Mapping[str, tuple[str, ...]]
    target_movements_by_source: Mapping[str, tuple[str, ...]]  # compatibility alias to routing_targets_by_source
    pressure_movement_by_movement: Mapping[str, str]
    pressure_group_members: Mapping[str, tuple[str, ...]]
    tls_link_from_lane: Mapping[tuple[str, int], str]

    def pressure_movement(self, movement_id: str) -> str:
        """Pressure representative for an actual controlled movement.

        V16 merges every controlled right turn into the straight movement of the
        same approach for pressure/state calculations.  The physical movement
        remains unchanged for routing and TraCI execution.  A T-approach with no
        straight movement is the only structural fallback and represents itself.
        """
        return self.pressure_movement_by_movement.get(movement_id, movement_id)

    def pressure_members(self, movement_id: str) -> tuple[str, ...]:
        rep = self.pressure_movement(movement_id)
        return self.pressure_group_members.get(rep, (rep,))

    def routing_movement(self, movement_id: str) -> RoutingMovementDef:
        return self.routing_movements[movement_id]

    def is_controlled_routing_movement(self, movement_id: str) -> bool:
        item = self.routing_movements.get(movement_id)
        return bool(item is not None and item.controlled_movement_id is not None)


class PhysicalEdgeGraph:
    """Small passenger edge graph extracted directly from the SUMO net XML.

    MP state lives on logical links; navigation and setRoute live on physical
    SUMO edges.  Keeping this graph separate prevents accidental edge/link
    conflation in the runtime.
    """

    def __init__(self, network_path: Path) -> None:
        root = ET.parse(network_path).getroot()
        edge_elements = {
            e.attrib["id"]: e
            for e in root.findall("edge")
            if not e.attrib["id"].startswith(":")
            and e.attrib.get("function") != "internal"
        }
        self.edge_length_m: dict[str, float] = {}
        self.edge_speed_mps: dict[str, float] = {}
        self.edge_from_junction: dict[str, str] = {}
        self.edge_to_junction: dict[str, str] = {}
        for edge_id, edge in edge_elements.items():
            lanes = edge.findall("lane")
            if not lanes:
                continue
            lengths = [float(l.attrib.get("length", 0.0)) for l in lanes]
            speeds = [float(l.attrib.get("speed", 0.0)) for l in lanes]
            length = sum(lengths) / max(len(lengths), 1)
            speed = max(speeds) if speeds else 0.0
            if length <= 0 or speed <= 0:
                raise ValueError(f"invalid edge geometry/speed for {edge_id}")
            self.edge_length_m[edge_id] = length
            self.edge_speed_mps[edge_id] = speed
            self.edge_from_junction[edge_id] = str(edge.attrib.get("from", ""))
            self.edge_to_junction[edge_id] = str(edge.attrib.get("to", ""))

        adjacency: dict[str, set[str]] = {e: set() for e in self.edge_length_m}
        reverse: dict[str, set[str]] = {e: set() for e in self.edge_length_m}
        connection_directions: dict[tuple[str, str], set[str]] = {}
        for conn in root.findall("connection"):
            source = conn.attrib.get("from")
            target = conn.attrib.get("to")
            if source in adjacency and target in adjacency:
                adjacency[source].add(target)
                reverse[target].add(source)
                direction = str(conn.attrib.get("dir", "")).strip()
                if direction:
                    connection_directions.setdefault((source, target), set()).add(direction)
        self.connection_directions = {k: tuple(sorted(v)) for k, v in connection_directions.items()}
        self.adjacency = {k: tuple(sorted(v)) for k, v in adjacency.items()}
        self.reverse = {k: tuple(sorted(v)) for k, v in reverse.items()}
        self.freeflow_time_s = {
            edge: self.edge_length_m[edge] / self.edge_speed_mps[edge]
            for edge in self.edge_length_m
        }

    def connected(self, a: str, b: str) -> bool:
        return b in self.adjacency.get(a, ())

    def validate_route(self, route: Iterable[str]) -> bool:
        edges = tuple(route)
        if not edges:
            return False
        return all(self.connected(a, b) for a, b in zip(edges, edges[1:]))

    def route_junction_sequence(self, route: Iterable[str]) -> tuple[str, ...]:
        """Return the physical junction sequence traversed by normal edges."""
        edges = tuple(route)
        if not edges:
            return ()
        if any(edge not in self.edge_from_junction for edge in edges):
            return ()
        return (self.edge_from_junction[edges[0]],) + tuple(
            self.edge_to_junction[edge] for edge in edges
        )

    def has_repeated_edge(self, route: Iterable[str]) -> bool:
        edges = tuple(route)
        return len(edges) != len(set(edges))

    def has_repeated_junction(self, route: Iterable[str]) -> bool:
        junctions = self.route_junction_sequence(route)
        return bool(junctions) and len(junctions) != len(set(junctions))

    def is_junction_simple_route(self, route: Iterable[str]) -> bool:
        edges = tuple(route)
        return bool(edges) and self.validate_route(edges) and not self.has_repeated_edge(edges) and not self.has_repeated_junction(edges)

    def route_time(self, route: Iterable[str], weights: Mapping[str, float]) -> float:
        total = 0.0
        for edge in route:
            value = float(weights.get(edge, self.freeflow_time_s.get(edge, float("inf"))))
            if not math.isfinite(value) or value <= 0:
                value = float(self.freeflow_time_s.get(edge, float("inf")))
            total += value
        return total

    def route_length(self, route: Iterable[str]) -> float:
        """Physical length of a route on the same edge sequence used by routing."""
        total = 0.0
        for edge in route:
            value = float(self.edge_length_m.get(edge, float("inf")))
            if not math.isfinite(value) or value <= 0:
                return float("inf")
            total += value
        return total

    def shortest_path(
        self,
        source: str,
        target: str,
        weights: Mapping[str, float] | None = None,
        *,
        forbidden: set[str] | None = None,
    ) -> tuple[str, ...] | None:
        if source not in self.adjacency or target not in self.adjacency:
            return None
        if source == target:
            return (source,)
        weights = self.freeflow_time_s if weights is None else weights
        forbidden = set() if forbidden is None else set(forbidden)
        forbidden.discard(source)
        forbidden.discard(target)
        pq: list[tuple[float, str]] = [(0.0, source)]
        dist = {source: 0.0}
        prev: dict[str, str] = {}
        while pq:
            cost, edge = heapq.heappop(pq)
            if cost != dist.get(edge):
                continue
            if edge == target:
                break
            for nxt in self.adjacency.get(edge, ()):
                if nxt in forbidden:
                    continue
                w = float(weights.get(nxt, self.freeflow_time_s.get(nxt, float("inf"))))
                if not math.isfinite(w) or w <= 0:
                    w = float(self.freeflow_time_s.get(nxt, float("inf")))
                new = cost + w
                if new < dist.get(nxt, float("inf")):
                    dist[nxt] = new
                    prev[nxt] = edge
                    heapq.heappush(pq, (new, nxt))
        if target not in dist:
            return None
        path = [target]
        while path[-1] != source:
            path.append(prev[path[-1]])
        path.reverse()
        return tuple(path)


def load_topology(topology_path: Path, network_path: Path) -> tuple[CVRouteTopology, PhysicalEdgeGraph]:
    graph = PhysicalEdgeGraph(network_path)
    raw = json.loads(topology_path.read_text(encoding="utf-8"))
    if raw.get("schema") not in {
        "lyg-clean-full-link-topology/v1",
        "lyg-cvroute-topology/v10-route-aware-controlled",
        "lyg-cvroute-topology/v14-fixed-physical-membership",
    }:
        raise ValueError("unexpected topology schema")

    # Physical edges may be grouped into one MP logical link.  Free-flow time is
    # therefore aggregated over all constituent physical edges, never copied
    # from one representative edge.
    links: dict[str, LinkDef] = {}
    edge_to_link: dict[str, str] = {}
    for link_id, item in raw["links"].items():
        edges = tuple(str(e) for e in item["edges"])
        if not edges:
            raise ValueError(f"logical link has no edges: {link_id}")
        ff = sum(graph.freeflow_time_s[e] for e in edges)
        for e in edges:
            previous = edge_to_link.get(e)
            if previous is not None and previous != link_id:
                raise ValueError(f"physical edge belongs to two logical links: {e}")
            edge_to_link[e] = link_id
        downstream_tls = str(item.get("downstream_tls", item["tls_id"]))
        upstream_tls_raw = str(item.get("upstream_tls", "")).strip()
        links[link_id] = LinkDef(
            link_id=link_id,
            tls_id=downstream_tls,
            upstream_tls=(upstream_tls_raw or None),
            downstream_tls=downstream_tls,
            boundary_upstream=bool(int(item.get("boundary_upstream", 0))),
            terminal_edge=str(item["terminal_edge"]),
            start_edge=str(item["start_edge"]),
            edges=edges,
            length_m=float(item["length_m"]),
            movement_ids=tuple(item["movement_ids"]),
            freeflow_time_s=ff,
        )

    movements: dict[str, MovementDef] = {}
    movement_by_tls_link_index: dict[tuple[str, int], str] = {}
    movement_by_link_to_edge: dict[tuple[str, str], str] = {}
    movement_by_edge_pair: dict[tuple[str, str], str] = {}
    for movement_id, item in raw["movements"].items():
        states = raw["phase_states"][str(item["tls_id"])]
        weights = {
            phase_id: sum(
                state[int(index)] in "gG" for index in item["link_indices"]
            )
            for phase_id, state in enumerate(states)
        }
        expected = {phase_id: count for phase_id, count in weights.items() if count}
        stored = {
            int(phase_id): int(count)
            for phase_id, count in item.get("phase_connection_weight", {}).items()
        }
        if set(map(int, item["served_phase_ids"])) != set(expected) or stored != expected:
            raise ValueError(
                f"Topology phase service does not match G/g signal states: {movement_id}"
            )
        movement = MovementDef(
            movement_id=movement_id,
            source_link=str(item["source_link"]),
            tls_id=str(item["tls_id"]),
            to_edge=str(item["to_edge"]),
            directions=tuple(str(v) for v in item.get("directions", ())),
            link_indices=tuple(int(v) for v in item["link_indices"]),
            served_phase_ids=tuple(int(v) for v in item["served_phase_ids"]),
            phase_connection_weight={
                int(k): int(v) for k, v in item.get("phase_connection_weight", {}).items()
            },
            next_link_candidates=tuple(str(v) for v in item.get("next_link_candidates", ())),
        )
        movements[movement_id] = movement
        movement_by_link_to_edge[(movement.source_link, movement.to_edge)] = movement_id
        terminal_edge = links[movement.source_link].terminal_edge
        edge_pair = (terminal_edge, movement.to_edge)
        previous_pair = movement_by_edge_pair.get(edge_pair)
        if previous_pair is not None and previous_pair != movement_id:
            raise ValueError(f"ambiguous controlled edge pair: {edge_pair}")
        movement_by_edge_pair[edge_pair] = movement_id
        for idx in movement.link_indices:
            key = (movement.tls_id, idx)
            previous = movement_by_tls_link_index.get(key)
            if previous is not None and previous != movement_id:
                raise ValueError(f"ambiguous TLS link index: {key}")
            movement_by_tls_link_index[key] = movement_id

    # Routing action space: every legal immediate exit from a fixed logical
    # link is a movement, whether or not that exit is controlled by a TLS.
    # Controlled movements reuse their existing IDs. Any EARLY exit before the
    # fixed-link terminal edge is the network's uncontrolled bypass/right-turn
    # movement and is explicitly retained as direction ``r``.
    def _norm_dir(values: tuple[str, ...] | list[str]) -> str:
        vals = {str(v).lower() for v in values}
        if "r" in vals:
            return "r"
        if "s" in vals:
            return "s"
        if "l" in vals:
            return "l"
        return "s"

    routing_movements: dict[str, RoutingMovementDef] = {}
    routing_movement_by_edge_pair: dict[tuple[str, str], str] = {}
    routing_movements_by_source_link_tmp: dict[str, list[str]] = {k: [] for k in links}

    # First register every controlled movement.
    for movement_id, movement in movements.items():
        pair = (links[movement.source_link].terminal_edge, movement.to_edge)
        direction = _norm_dir(list(movement.directions) + list(graph.connection_directions.get(pair, ())))
        routing_movements[movement_id] = RoutingMovementDef(
            movement_id=movement_id,
            source_link=movement.source_link,
            from_edge=pair[0],
            to_edge=pair[1],
            direction=direction,
            controlled_movement_id=movement_id,
            freeflow_time_s=float(links[movement.source_link].freeflow_time_s),
        )
        routing_movement_by_edge_pair[pair] = movement_id
        routing_movements_by_source_link_tmp[movement.source_link].append(movement_id)

    # Then add legal exits that are not in the controlled topology.
    for link_id, link in links.items():
        link_edges = set(link.edges)
        for edge in link.edges:
            for nxt in graph.adjacency.get(edge, ()):
                if nxt in link_edges:
                    continue
                pair = (edge, nxt)
                if pair in routing_movement_by_edge_pair:
                    continue
                dirs = graph.connection_directions.get(pair, ())
                # In this fixed-link representation an exit before the terminal
                # edge is exactly the uncontrolled bypass/right-turn branch, even
                # when SUMO's geometric connection label happens to be ``s``.
                direction = "r" if edge != link.terminal_edge else _norm_dir(list(dirs))
                if direction not in {"l", "s", "r"}:
                    continue
                movement_id = f"U:{link_id}:{edge}->{nxt}"
                exit_idx = link.edges.index(edge)
                movement_ff = sum(graph.freeflow_time_s[e] for e in link.edges[: exit_idx + 1])
                routing_movements[movement_id] = RoutingMovementDef(
                    movement_id=movement_id,
                    source_link=link_id,
                    from_edge=edge,
                    to_edge=nxt,
                    direction=direction,
                    controlled_movement_id=None,
                    freeflow_time_s=float(movement_ff),
                )
                routing_movement_by_edge_pair[pair] = movement_id
                routing_movements_by_source_link_tmp[link_id].append(movement_id)

    routing_movements_by_source_link = {
        link_id: tuple(dict.fromkeys(mids))
        for link_id, mids in routing_movements_by_source_link_tmp.items()
    }
    routing_targets_by_source: dict[str, tuple[str, ...]] = {}
    for source_id, movement in movements.items():
        targets: list[str] = []
        for link_id in movement.next_link_candidates:
            targets.extend(routing_movements_by_source_link.get(link_id, ()))
        routing_targets_by_source[source_id] = tuple(dict.fromkeys(targets))
    # Compatibility alias used by a few diagnostics; semantics are now ALL
    # immediate routing movements, not only controlled downstream movements.
    target_movements_by_source = dict(routing_targets_by_source)

    # V16 pressure grouping: a controlled right turn contributes to the
    # straight movement of the SAME source approach. This is a pressure/state
    # aggregation only; actual movement identity is retained for routing,
    # passage counting, route feasibility and TraCI execution. Uncontrolled
    # bypass/right-turn states are represented separately in routing_movements
    # and are never merged into a downstream controlled movement.
    movements_by_source: dict[str, list[str]] = {}
    for movement_id, movement in movements.items():
        movements_by_source.setdefault(movement.source_link, []).append(movement_id)
    pressure_movement_by_movement: dict[str, str] = {}
    for source_link, mids in movements_by_source.items():
        straight = [m for m in mids if "s" in movements[m].directions]
        if len(straight) > 1:
            raise ValueError(f"multiple straight movements on one source link: {source_link}: {straight}")
        straight_id = straight[0] if straight else None
        for movement_id in mids:
            movement = movements[movement_id]
            if "r" in movement.directions and straight_id is not None:
                pressure_movement_by_movement[movement_id] = straight_id
            else:
                pressure_movement_by_movement[movement_id] = movement_id
    pressure_group_members_tmp: dict[str, list[str]] = {}
    for movement_id, rep in pressure_movement_by_movement.items():
        pressure_group_members_tmp.setdefault(rep, []).append(movement_id)
    pressure_group_members = {
        rep: tuple(sorted(mids)) for rep, mids in pressure_group_members_tmp.items()
    }


    # Physical TLS link-index -> source lane mapping used by the corrected
    # movement service-capacity calculation.  A physical fromLane is a shared
    # service resource even when SUMO exposes multiple turning connections from
    # that lane.  The mapping is network-derived and contains no demand state.
    net_root = ET.parse(network_path).getroot()
    tls_link_from_lane: dict[tuple[str, int], str] = {}
    for conn in net_root.findall("connection"):
        tls_id = conn.attrib.get("tl")
        link_index = conn.attrib.get("linkIndex")
        from_edge = conn.attrib.get("from")
        from_lane = conn.attrib.get("fromLane")
        if tls_id is None or link_index is None or from_edge is None or from_lane is None:
            continue
        key = (str(tls_id), int(link_index))
        lane_id = f"{from_edge}_{from_lane}"
        previous = tls_link_from_lane.get(key)
        if previous is not None and previous != lane_id:
            raise ValueError(f"TLS link index maps to multiple source lanes: {key}: {previous}, {lane_id}")
        tls_link_from_lane[key] = lane_id

    topology = CVRouteTopology(
        links=links,
        movements=movements,
        phase_states={k: tuple(v) for k, v in raw["phase_states"].items()},
        movement_by_tls_link_index=movement_by_tls_link_index,
        movement_by_link_to_edge=movement_by_link_to_edge,
        movement_by_edge_pair=movement_by_edge_pair,
        edge_to_link=edge_to_link,
        routing_movements=routing_movements,
        routing_movement_by_edge_pair=routing_movement_by_edge_pair,
        routing_movements_by_source_link=routing_movements_by_source_link,
        routing_targets_by_source=routing_targets_by_source,
        target_movements_by_source=target_movements_by_source,
        pressure_movement_by_movement=pressure_movement_by_movement,
        pressure_group_members=pressure_group_members,
        tls_link_from_lane=tls_link_from_lane,
    )
    return topology, graph
