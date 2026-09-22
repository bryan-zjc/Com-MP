from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExperimentConfig:
    experiment_name: str = "com_mp_5x5_d15k_cv08"
    intersection: str = "5x5"
    max_steps: int = 3600
    warm_start_steps: int = 150
    decision_interval_seconds: int = 10
    episodes: int = 100
    cv_penetration_rate: float = 0.8
    seed: int = 20260703
    gui: bool = False

    # Alpha keeps the same meaning in every state. No state-dependent mask is
    # applied; behaviorally irrelevant decisions are excluded from actor loss.
    alpha_bins: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    # Per-second local reward:
    # omega_1 * sum(movement queue-ratio reductions)
    # - omega_2 * sum(current movement queue ratios).
    queue_reduction_reward_weight: float = 1.0
    queue_level_reward_weight: float = 0.10
    reward_scale: float = 1.0
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

    # W&B is optional. Set WANDB_API_KEY and pass --wandb to enable it.
    use_wandb: bool = False
    wandb_project: str = "com_mp_mappo"

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
        return (ROOT / "outputs" / self.intersection /
                self.experiment_name)

    @property
    def checkpoint_dir(self):
        return (ROOT / "checkpoints" / self.intersection /
                self.experiment_name)
