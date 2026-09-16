This is the demo code for our paper in the 5x5 grid network: **Com-MP: Heterogeneity-Aware Composite Max-Pressure Control for Urban Road Networks**. The public example includes only the synthetic 5x5 road network and its demand file. No trained checkpoints, experiment results, Qinzhou data, or personal credentials are included.

## Abstract
Max-pressure (MP) control provides a decentralized and scalable framework for network-level traffic signal control with explicit queue-stability properties. However, different MP formulations emphasize different traffic conditions and rely on different sensing sources, making a fixed pressure representation difficult to maintain under heterogeneous and time-varying urban traffic environments. This paper proposes Com-MP, a reinforcement-learning-enhanced framework that adaptively composes multiple MP control preferences while retaining decentralized MP phase selection. At each decision epoch, each constituent controller first evaluates the candidate phases using its own MP formulation, and the resulting phase pressures are normalized before an intersection-specific weighted composition is formed. A shared policy maps local traffic and sensing observations to these composition weights, allowing different intersections to adapt independently while using the same topology-independent decision rule. Because learning is confined to pressure composition rather than signal actions, the resulting controller remains structurally linked to the underlying MP mechanism. We further show that the resulting intersection-wise adaptive composition preserves strong queue stability over the strictly stabilizable demand region associated with the same finite-storage traffic dynamics and operational signal constraints. Experiments on a real-world network with 45 signalized intersections show that Com-MP reduces average delay by approximately 40.0% and 15.0% relative to queue-based MP and CV-based travel-time MP, respectively, while achieving the highest average speed and throughput. The controller also maintains strong performance under varying traffic demand, detector coverage, and CV penetration. Moreover, the learned policy can be deployed without retraining on an unseen network and achieves performance close to a policy trained directly on the target network.

<img width="4815" height="1978" alt="framework" src="https://github.com/user-attachments/assets/90b1e891-d1d9-4697-8816-40033e72fac2" />


_Fig.1 Framework of the proposed Com-MP controller. The upper layer adapts the composition of constituent MP preferences, while the lower layer retains decentralized MP phase selection._

## Repository structure

```text
Com-MP-GitHub/
|-- envs/
|   `-- 5x5/
|       |-- 5x5.net.xml
|       |-- 5x5.sumocfg
|       `-- Demand.rou.xml
|-- src/
|   |-- road_env/
|   |   |-- env_MP.py
|   |   |-- env_CVMP.py
|   |   `-- env_hybrid_MP.py
|   `-- utils/
|       `-- utils.py
|-- training/
|   |-- config_v6.py
|   |-- env_hybrid_mappo.py
|   |-- env_hybrid_mappo_v6.py
|   |-- hybrid_mappo_agent.py
|   `-- train_hybrid_mappo_v6.py
|-- tripinfo_statistics.py
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.10 or 3.11
- Eclipse SUMO with `SUMO_HOME` configured
- NumPy 1.x (`numpy>=1.24,<2`)
- PyTorch

## Training

From the repository root, run:

```bash
python training/train_hybrid_mappo_v6.py
```

The default experiment uses:

- network: `5x5`;
- route file: `envs/5x5/Demand.rou.xml`;
- episode horizon: 1,800 seconds;
- fixed-time warm start: 150 seconds;
- decision interval: 10 seconds;
- CV penetration rate: 0.5;
- training episodes: 100;
- random seed: 20260703.

Useful command-line options include:

```bash
python training/train_hybrid_mappo_v6.py \
  --episodes 100 \
  --max-steps 1800 \
  --seed 20260703 \
  --cv-rate 0.5 \
  --experiment-name com_mp_5x5
```

Generated histories and tripinfo files are written to
`outputs/5x5/com_mp_5x5/`. Checkpoints are written to
`checkpoints/5x5/com_mp_5x5/`. These directories are ignored by Git.


## Data and outputs

Only the 5x5 example network and demand are distributed. The repository does not include training outputs, checkpoints, W&B files, or results from other networks. Users are responsible for ensuring that any additional network or demand data they add can be redistributed.

Due to data-sensitivity considerations, detailed information and files for the Qinzhou road network are not publicly released at this time. Researchers with a need for access are welcome to contact the authors for further information.

