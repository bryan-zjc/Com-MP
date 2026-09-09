# Com-MP: Heterogeneity-Aware Composite Max-Pressure Control for Urban Road Networks

This is the demo code for our paper: **Com-MP: Heterogeneity-Aware Composite Max-Pressure Control for Urban Road Networks**.

## Abstract
Max-pressure (MP) control provides a decentralized and scalable framework for network-level traffic signal control with explicit queue-stability properties. However, different MP formulations emphasize different traffic conditions and rely on different sensing sources, making a fixed pressure representation difficult to maintain under heterogeneous and time-varying urban traffic environments. This paper proposes Com-MP, a reinforcement-learning-enhanced framework that adaptively composes multiple MP control preferences while retaining decentralized MP phase selection. At each decision epoch, a shared policy maps local traffic and sensing observations to an intersection-specific composition of constituent MP pressures, allowing different intersections to adapt independently while using the same topology-independent decision rule. Because learning is confined to pressure composition rather than signal actions, the resulting controller remains structurally linked to the underlying MP mechanism. We further show that the resulting intersection-wise adaptive composition preserves strong queue stability under admissible demand. Experiments on a real-world network with 45 signalized intersections show that Com-MP reduces average delay by approximately 40.0% and 15.0% relative to queue-based MP and CV-based travel-time MP, respectively, while achieving the highest average speed and throughput. The controller also maintains strong performance under varying traffic demand, detector coverage, and CV penetration. Moreover, the learned policy can be deployed without retraining on an unseen network and achieves performance close to a policy trained directly on the target network.

<img width="4815" height="1978" alt="framework" src="https://github.com/user-attachments/assets/276ab74b-ab7e-41b3-94ac-5ed2515c92d9" />
_Fig.1 Framework of the proposed Com-MP controller. The upper layer adapts the composition of constituent MP preferences, while the lower layer retains decentralized MP phase selection._

## Requirements

- Python 3.10 or newer
- Eclipse SUMO with its Python tools
- PyTorch

## Training

Run commands from the repository root.

Train on the Qinzhou network:

```bash
python agent/train_hybrid_mappo_v6.py \
  --intersection qinzhou \
  --episodes 300 \
  --max-steps 1800 \
  --cv-rate 0.5 \
  --no-wandb
```

Train on the 5x5 grid:

```bash
python agent/train_hybrid_mappo_v6.py \
  --intersection 5x5 \
  --episodes 300 \
  --max-steps 1800 \
  --cv-rate 0.5 \
  --no-wandb
```

The main command-line options are:

| Option | Description | Default |
|---|---|---:|
| `--intersection` | Training network: `qinzhou` or `5x5` | `qinzhou` |
| `--episodes` | Number of training episodes | `300` |
| `--max-steps` | SUMO simulation horizon per episode | `1800` |
| `--seed` | Training and SUMO random seed | `20260703` |
| `--cv-rate` | Global CV penetration rate | `0.5` |
| `--switch-margin` | Pressure margin required for a phase switch | `0.05` |
| `--validation-interval` | Episodes between deterministic validations | `20` |
| `--resume` | Checkpoint path for resuming training | empty |
| `--wandb` / `--no-wandb` | Enable or disable W&B logging | disabled |

The first 150 seconds of each simulation are a fixed-time warm-start period and are excluded from training. The current implementation intentionally uses the same SUMO seed in every training episode for a fixed-demand training setting. This behavior can be changed in `agent/train_hybrid_mappo_v6.py` if randomized episode demands are required.

## Outputs

Training creates local artifacts only when the program is run:

```text
outputs/<network>/<experiment-name>/
checkpoints/<network>/<experiment-name>/
```

These directories are excluded by `.gitignore`. `latest.pt` is updated after every episode, while `best.pt` is updated when deterministic validation produces a new minimum average delay.

## Network data

Each supported network directory must contain:

```text
envs/<network>/<network>.sumocfg
envs/<network>/<network>.net.xml
envs/<network>/Demand.rou.xml
```

To add another network, follow this layout and add its name to `SUPPORTED_INTERSECTIONS` in `agent/train_hybrid_mappo_v6.py`.

## Reproducibility notes

- The Python, NumPy, PyTorch, SUMO, and CV-assignment seeds are initialized from `--seed`.
- Validation uses fixed seeds defined in `agent/config_v6.py`.
- Pressure vectors are normalized independently at each intersection before composition.
- The code uses `LIBSUMO_AS_TRACI=1` by default to avoid socket communication overhead.
