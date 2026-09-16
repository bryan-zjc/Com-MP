# Com-MP: Composite Max-Pressure Traffic Signal Control

This is the demo code of our paper: **Com-MP: Heterogeneity-Aware Composite Max-Pressure Control for Urban Road Networks**

The public example includes only the synthetic 5x5 road network and its demand file. 

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

Only the 5x5 example network and demand are distributed. The repository does
not include training outputs, checkpoints, W&B files, or results from other
networks. Users are responsible for ensuring that any additional network or
demand data they add can be redistributed.

Due to data-sensitivity considerations, detailed information and files for the
Qinzhou road network are not publicly released at this time. Researchers with
a legitimate need for access are welcome to contact the authors for further
information.
