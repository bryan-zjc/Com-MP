from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExperimentConfig:
    experiment_name: str = "com_mp_v0909"
    intersection: str = "5x5"
    max_steps: int = 1800
    warm_start_steps: int = 150
    decision_interval_seconds: int = 10
    episodes: int = 300
    cv_penetration_rate: float = 0.5
    seed: int = 20260703
    gui: bool = False

    # Alpha keeps the same meaning in every state. No state-dependent mask is
    # applied; behaviorally irrelevant decisions are excluded from actor loss.
    alpha_bins: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    # Detector-only reward: reward queue discharge immediately, while retaining
    # a small queue-level term so the objective does not telescope to the final
    # queue alone. Both terms use fixed roadside detector measurements.
    queue_reduction_reward_weight: float = 1.0
    queue_level_reward_weight: float = 0.10
    queue_change_clip: float = 0.0  # Disabled; positive values are queue-ratio changes.
    reward_scale: float = 1.0
    max_red_seconds: float = 180.0
    gamma_per_second: float = 0.997
    gae_lambda_per_second: float = 0.97
    learning_rate: float = 3e-4
    learning_rate_final: float = 3e-5
    ppo_clip: float = 0.2
    value_coef: float = 0.5
    entropy_initial: float = 0.015
    entropy_final: float = 0.001
    update_epochs: int = 5
    minibatch_size: int = 512
    max_grad_norm: float = 0.5
    hidden_size: int = 256
    target_kl: float = 0.02

    validation_interval: int = 20
    validation_seeds: tuple = (20260801, 20260802, 20260803)

    use_wandb: bool = False
    wandb_project: str = "hybrid_MP_MAPPO"

    @property
    def sumocfg_file(self):
        return ROOT / "envs" / self.intersection / f"{self.intersection}.sumocfg"

    @property
    def net_file(self):
        return ROOT / "envs" / self.intersection / f"{self.intersection}.net.xml"

    @property
    def rou_file(self):
        return ROOT / "envs" / self.intersection / "Demand.rou.xml"

    @property
    def output_dir(self):
        return (ROOT / "com_mp" / "results" / self.intersection /
                self.experiment_name)

    @property
    def checkpoint_dir(self):
        return (ROOT / "com_mp" / "checkpoints" / self.intersection /
                self.experiment_name)
