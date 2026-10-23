# Com-MP: Composite Max-Pressure Traffic Signal Control

This is the demo code of our paper: **Com-MP: Heterogeneity-Aware Composite Max-Pressure Control for Urban Road Networks**.

## Abstract

Reinforcement learning (RL) provides a flexible mechanism for adapting traffic signal control to heterogeneous traffic and sensing conditions, but directly learned signal actions do not automatically inherit the analytical properties of structured traffic-control methods. This paper proposes Com-MP, a theory-guided RL framework that learns a context-dependent composition of multiple max-pressure (MP) control preferences while retaining decentralized MP phase selection. The learned policy adapts the relative influence of constituent MP formulations according to local traffic and sensing conditions, allowing data-driven adaptation to operate within a structured traffic-control framework. We show that such intersection-specific and time-varying composition preserves strong queue stability for every demand in the strictly stabilizable region under any admissible non-anticipative composition policy, thereby separating the performance-adaptation role of learning from the stability property of the underlying controller. In a representative implementation combining detector-supported queue-based MP and connected-vehicle-supported travel-time MP, experiments on a real-world 45-intersection network show that Com-MP reduces average delay by 15.6\% and 8.6\% relative to the two constituent controllers, respectively, and by about 7.0\% relative to the best fixed composition. The learned composition responds systematically to local traffic and sensing conditions, remains effective under heterogeneous and degraded sensing, and can be transferred without retraining to an unseen network with an average delay only 2.4\% higher than that of a policy trained directly on the target network. These results show how learning-based adaptation can be introduced within a stability-preserving traffic-control structure, providing a theory-guided pathway toward trustworthy learning-enabled traffic signal control.

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

