# Com-MP: Composite Max-Pressure Traffic Signal Control

This is the demo code of our paper: **Com-MP: Heterogeneity-Aware Composite Max-Pressure Control for Urban Road Networks**

The public example includes only the synthetic 5x5 road network and its demand
file. No trained checkpoints, experiment results, Qinzhou data, or personal
credentials are included.

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

## Main implementation

The implementation follows these operational choices:

- Q-MP uses the movement vehicle count divided by incoming-lane length.
- Turning ratios are estimated from realized local turning flows over the most
  recent 100 seconds.
- Movement service is set to zero when the head vehicle cannot use the
  candidate phase or the downstream link cannot receive traffic.
- A phase switch applies a 0.7 service multiplier for a 10-second decision
  interval with 3 seconds of yellow time.
- A phase is forced when its red time reaches 180 seconds.
- The Com-MP action space is `alpha = {0, 0.25, 0.5, 0.75, 1}`.
- The MAPPO actor is shared across intersections, while training uses a
  centralized critic and decentralized execution.
- Standard four-leg intersections use eight movement slots and four phase
  slots. Missing slots can be zero-padded and excluded by the topology-mask
  interface.

The local observation contains normalized Q-MP phase pressures, normalized
CV-MP phase pressures, movement queue-occupancy ratios, the current phase
one-hot vector, elapsed green time, and the local CV ratio.

The per-second local reward is

```text
queue_reduction_weight * sum(movement queue-ratio reductions)
- queue_level_weight * sum(current movement queue ratios)
```

## Requirements

- Python 3.10 or 3.11
- Eclipse SUMO with `SUMO_HOME` configured
- NumPy 1.x (`numpy>=1.24,<2`)
- PyTorch

Create and activate an isolated environment before installing dependencies.
For example:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export SUMO_HOME=/path/to/sumo
```

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

## Weights & Biases

W&B logging is disabled by default and no API key is stored in this
repository. To enable it, set your own key and pass `--wandb`:

```powershell
$env:WANDB_API_KEY = "YOUR_WANDB_API_KEY"
python training/train_hybrid_mappo_v6.py --wandb
```

Use `--no-wandb` to explicitly disable online logging.

## Reproducibility notes

All training episodes use the configured seed by default, so the SUMO demand
realization and CV assignment remain fixed across episodes. Validation uses the
fixed seeds declared in `training/config_v6.py`. Change these values when
evaluating robustness across independent random realizations.

## Data and outputs

Only the 5x5 example network and demand are distributed. The repository does
not include training outputs, checkpoints, W&B files, or results from other
networks. Users are responsible for ensuring that any additional network or
demand data they add can be redistributed.

Due to data-sensitivity considerations, detailed information and files for the
Qinzhou road network are not publicly released at this time. Researchers with
a legitimate need for access are welcome to contact the authors for further
information.
