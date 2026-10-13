from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass(slots=True)
class CVMPConfig:
    sumo_seed: int = 1
    connected_penetration: float = 0.50
    control_interval_s: int = 15
    yellow_s: int = 3
    yellow_service_factor: float = 0.0
    maximum_red_s: int = 180
    saturation_flow_vph_per_lane: float = 1800.0
    turning_window_s: int = 300
    service_failure_feedback_enabled: bool = True
    service_failure_required_intervals: int = 2
    service_failure_min_waiting_connected: int = 2
    service_failure_waiting_speed_mps: float = 0.1
    maximum_simulation_s: int = 9000
    demand_end_s: int = 9000
    gui: bool = False
    show_progress: bool = True
    save_tripinfo: bool = True
    cv_color: str = "0,102,204"
    nv_color: str = "128,128,128"

    @property
    def switch_service_factor(self) -> float:
        return ((self.control_interval_s - self.yellow_s) + self.yellow_s * self.yellow_service_factor) / self.control_interval_s

    def validate(self) -> None:
        if self.control_interval_s != 15 or self.yellow_s != 3:
            raise ValueError('v5 requires 15-second decisions and 3-second yellow')
        if not 0.0 <= self.connected_penetration <= 1.0:
            raise ValueError("connected_penetration must lie in [0,1]")
        if self.control_interval_s <= 0 or self.yellow_s < 0 or self.yellow_s >= self.control_interval_s:
            raise ValueError("invalid control/yellow timing")
        if self.maximum_red_s <= 0:
            raise ValueError("maximum_red_s must be positive")
        if self.saturation_flow_vph_per_lane <= 0:
            raise ValueError("saturation flow must be positive")
        if self.turning_window_s <= 0:
            raise ValueError("turning_window_s must be positive")
        if self.service_failure_required_intervals != 2:
            raise ValueError("service_failure_required_intervals must be 2")
        if self.maximum_simulation_s <= 0:
            raise ValueError("maximum_simulation_s must be positive")

    def to_json_dict(self):
        d=asdict(self)
        d["switch_service_factor"]=self.switch_service_factor
        return d

def default_paths(root: Path, intersection: str = "qinzhou") -> dict[str, Path]:
    return {
        "network": root / "envs" / intersection / f"{intersection}.net.xml",
        "topology": root / "envs" / intersection / "topology.json",
        "route": root / "envs" / intersection / "Demand.rou.xml",
    }
