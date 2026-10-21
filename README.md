# Com-MP: Composite Max-Pressure Traffic Signal Control

This is the demo code of our paper: **Com-MP: Heterogeneity-Aware Composite Max-Pressure Control for Urban Road Networks**.

## Abstract

Max-pressure (MP) control provides a decentralized and scalable framework for network-level traffic signal control with explicit queue-stability properties. However, different MP formulations emphasize different congestion characteristics and rely on different sensing sources, so their relative usefulness can vary with local traffic states and sensing environments. This paper proposes Com-MP, a framework that learns a context-dependent composition of multiple MP control preferences while retaining decentralized MP phase selection. At each decision epoch, constituent controllers evaluate candidate phases using their own pressure formulations, and their normalized phase preferences are combined using intersection-specific weights. A shared reinforcement-learning policy learns these weights from local traffic and sensing observations, allowing the composition to adapt across intersections and over time without directly learning signal phases. We show that such intersection-specific and time-varying composition preserves strong queue stability for every demand in the strictly stabilizable region. In a representative implementation combining detector-supported queue-based MP and CV-supported travel-time MP, experiments on a real-world 45-intersection network show that Com-MP reduces average delay by 15.6\% and 8.6\% relative to the two constituent controllers, respectively. It further reduces delay by about 7.0\% relative to the best fixed composition, demonstrating the benefit of context-dependent adaptation. The learned composition responds systematically to local traffic and sensing conditions and remains effective under varying detector and CV availability. When transferred without retraining to an unseen network, it achieves an average delay only 2.4\% higher than a policy trained directly on the target network.

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

