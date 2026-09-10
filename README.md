This is the demo code for our paper in the 5x5 grid network: **Com-MP: Heterogeneity-Aware Composite Max-Pressure Control for Urban Road Networks**.

## Abstract
Max-pressure (MP) control provides a decentralized and scalable framework for network-level traffic signal control with explicit queue-stability properties. However, different MP formulations emphasize different traffic conditions and rely on different sensing sources, making a fixed pressure representation difficult to maintain under heterogeneous and time-varying urban traffic environments. This paper proposes Com-MP, a reinforcement-learning-enhanced framework that adaptively composes multiple MP control preferences while retaining decentralized MP phase selection. At each decision epoch, each constituent controller first evaluates the candidate phases using its own MP formulation, and the resulting phase pressures are normalized before an intersection-specific weighted composition is formed. A shared policy maps local traffic and sensing observations to these composition weights, allowing different intersections to adapt independently while using the same topology-independent decision rule. Because learning is confined to pressure composition rather than signal actions, the resulting controller remains structurally linked to the underlying MP mechanism. We further show that the resulting intersection-wise adaptive composition preserves strong queue stability over the strictly stabilizable demand region associated with the same finite-storage traffic dynamics and operational signal constraints. Experiments on a real-world network with 45 signalized intersections show that Com-MP reduces average delay by approximately 40.0% and 15.0% relative to queue-based MP and CV-based travel-time MP, respectively, while achieving the highest average speed and throughput. The controller also maintains strong performance under varying traffic demand, detector coverage, and CV penetration. Moreover, the learned policy can be deployed without retraining on an unseen network and achieves performance close to a policy trained directly on the target network.

<img width="4815" height="1978" alt="framework" src="https://github.com/user-attachments/assets/90b1e891-d1d9-4697-8816-40033e72fac2" />


_Fig.1 Framework of the proposed Com-MP controller. The upper layer adapts the composition of constituent MP preferences, while the lower layer retains decentralized MP phase selection._

## Requirements

- Python 3.10 or newer
- Eclipse SUMO with its Python tools
- PyTorch

## Training

```bash
python "com_mp/train_hybrid_mappo_v6.py" --experiment-name grid_run_01 --no-wandb
```

Training creates:

```text
com_mp/results/5x5/grid_run_01/
  training_history.csv
  train_ep001.xml
  validation/
com_mp/checkpoints/5x5/grid_run_01/
  latest.pt
  best.pt
```

Use a new experiment name for each fresh training run: histories are appended
and model files can be overwritten. Neither directory is shipped in this repository.
Resume only compatible checkpoints, using `--resume PATH`. A 30-feature
checkpoint is not compatible with this 18-feature implementation.

The first 150 seconds of each simulation are a fixed-time warm-start period and are excluded from training. The current implementation intentionally uses the same SUMO seed in every training episode for a fixed-demand training setting. This behavior can be changed in `agent/train_hybrid_mappo_v6.py` if randomized episode demands are required.

## Evaluate

After training:

```bash
python "com_mp/evaluate_v6.py" --experiment-name grid_run_01 --checkpoint "com_mp/checkpoints/5x5/grid_run_01/latest.pt" --methods MP,CVMP,HybridMAPPO --seeds 20260801,20260802,20260803
```

For baselines without a checkpoint, use `--methods MP,CVMP`.
`HybridMAPPO` is the script identifier for Com-MP.
Evaluation writes its own files under the selected experiment directory.


## Files

```text
com_mp/
  train_hybrid_mappo_v6.py  # MAPPO training and validation
  config_v6.py             # Settings and portable paths
  hybrid_mappo_agent.py    # Actor, critic and grouped PPO
  env_hybrid_mappo_v6.py   # Active observation, reward and signal controller
  env_hybrid_mappo.py      # Simulation caches and shared environment support
  evaluate_v6.py           # Baseline and learned-policy evaluation
src/road_env/             # Required inherited MP/CV-MP environment definitions
src/utils/utils.py        # Required SUMO/network helpers
envs/5x5/                 # Network, demand and SUMO configuration
tripinfo_statistics.py    # Trip-level metrics
```

