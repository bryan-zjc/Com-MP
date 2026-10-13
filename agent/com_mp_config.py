from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ExperimentConfig:
    experiment_name: str = "com_mp_v9_phase_context_dynamic_3h"
    intersection: str = "qinzhou"
    max_steps: int = 10800
    warm_start_steps: int = 0
    decision_interval_seconds: int = 15
    episodes: int = 100
    cv_penetration_rate: float = 0.5
    seed: int = 1
    gui: bool = False
    dynamic_training_scenario: bool = True
    dynamic_training_profile: str = "11k_15k_13k_cv020403"
    dynamic_demand_veh_per_h: tuple = (11000, 15000, 13000)
    dynamic_cv_penetration_rates: tuple = (0.20, 0.40, 0.30)

    # Alpha keeps the same meaning in every state. No state-dependent mask is
    # applied; behaviorally irrelevant decisions are excluded from actor loss.
    alpha_bins: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    # Reward uses the total halting vehicles on complete incoming links.
    queue_reduction_reward_weight: float = 1.0
    queue_level_reward_weight: float = 0.10
    reward_scale: float = 1.0
    gamma_per_second: float = 0.997
    gae_lambda: float = 0.97
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
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

    # W&B is enabled by default; pass --no-wandb to disable it.
    use_wandb: bool = True
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
    def dynamic_training_dir(self):
        return (ROOT / "envs" / "qinzhou" /
                "dynamic_3h_11k_15k_13k_training")

    @property
    def output_dir(self):
        return (ROOT / "outputs" / self.intersection /
                self.experiment_name)

    @property
    def checkpoint_dir(self):
        return (ROOT / "checkpoints" / self.intersection /
                self.experiment_name)
