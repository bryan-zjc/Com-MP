# Com-MP: Composite Max-Pressure Traffic Signal Control

This is the demo code of our paper: **Com-MP: Heterogeneity-Aware Composite Max-Pressure Control for Urban Road Networks**.

## Abstract

Max-pressure (MP) control provides a decentralized and scalable framework for network-level traffic signal control with explicit queue-stability properties. However, different MP formulations emphasize different traffic conditions and rely on different sensing sources, making a fixed pressure representation difficult to maintain under heterogeneous and time-varying urban traffic environments. This paper proposes Com-MP, a reinforcement-learning-enhanced framework that adaptively composes multiple MP control preferences while retaining decentralized MP phase selection. At each decision epoch, each constituent controller first evaluates the candidate phases using its own MP formulation, and the resulting phase pressures are normalized before an intersection-specific weighted composition is formed. A shared policy maps local traffic and sensing observations to these composition weights, allowing different intersections to adapt independently while using the same topology-independent decision rule. Because learning is confined to pressure composition rather than signal actions, the resulting controller remains structurally linked to the underlying MP mechanism. We further show that the resulting intersection-wise adaptive composition preserves strong queue stability over the strictly stabilizable demand region associated with the same finite-storage traffic dynamics and operational signal constraints. Experiments on a real-world network with 45 signalized intersections show that Com-MP reduces average delay by approximately 15.6\% and 8.6\% relative to queue-based MP and CV-based travel-time MP, respectively, while achieving the highest average speed and throughput. The controller also maintains strong performance under varying traffic demand, detector coverage, and CV penetration. Moreover, the learned policy can be deployed without retraining on an unseen network and achieves performance close to a policy trained directly on the target network.

<img width="4815" height="1978" alt="framework" src="https://github.com/user-attachments/assets/cae06d1b-1b76-41aa-a97e-1475f67630e5" />

_Fig.1 Framework of the proposed Com-MP controller. The upper layer adapts the composition of normalized constituent phase pressures, while the lower layer retains decentralized MP phase selection._

## Structure
This repository is the Qinzhou core implementation. It contains:

- Q-MP controller: `agent/main_MP.py`
- CV-MP controller: `agent/main_CVMP.py`
- Com-MP training: `agent/train_com_mp.py`
- Shared Com-MP/MAPPO and SUMO environment modules
- Qinzhou network, topology, and training demand files

## Requirements

- Python 3.10 or 3.11
- SUMO, with `SUMO_HOME` pointing to the SUMO installation
- A PyTorch build suitable for the local CPU or CUDA environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Weights & Biases is optional. Use `wandb login` or set `WANDB_API_KEY` in the environment; the repository contains no credentials.

## Run Q-MP

```bash
python agent/main_MP.py --help
```

## Run CV-MP

```bash
python agent/main_CVMP.py --help
```

## Train Com-MP

The default Qinzhou training profile has three one-hour demand stages of 11,000, 15,000, and 13,000 veh/h.

```bash
python agent/train_com_mp.py --episodes 100 --no-wandb
```

To enable W&B logging, replace `--no-wandb` with `--wandb`.

