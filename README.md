# Com-MP: Heterogeneity-Aware Composite Max-Pressure Control for Urban Road Networks

This is the demo code for our paper: **Com-MP: Heterogeneity-Aware Composite Max-Pressure Control for Urban Road Networks**.

## Abstract
Max-pressure (MP) control is a promising approach for network-level traffic signal control due to its decentralized implementation, scalability, and stability guarantees. However, existing MP variants favor different traffic conditions, and a single fixed pressure formulation is often insufficient for complex and time-varying urban networks. To address this issue, this paper proposes a reinforcement-learning-enhanced compositional MP control framework to improve the adaptability of MP-based signal control. The proposed framework adopts a two-layer architecture. At the lower level, a generalized MP controller is developed by constructing the pressure state as a weighted combination of stability-preserving atomic states, so that different operational preferences can be unified within a common MP framework. A corresponding parameterized Lyapunov function is established to prove the stability of the resulting controller. At the upper level, a reinforcement learning agent operates on a slower time scale to fine-tune the weighting coefficients within a stability-preserving safe set, enabling the controller to adapt its pressure preference to varying traffic conditions while improving delay- and stop-related performance. The proposed method is evaluated on a traffic network with 48 signalized intersections under multiple traffic demand scenarios. Numerical results show that the proposed framework consistently outperforms single-formulation MP controllers and exhibits stronger adaptability to heterogeneous and dynamically changing traffic conditions. In particular, the proposed method is able to preserve efficient queue dissipation under highly unbalanced demands, while achieving better overall delay and stop performance in more balanced and time-varying traffic scenarios.

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
- Experimental outputs and pretrained models are intentionally omitted from this release.
