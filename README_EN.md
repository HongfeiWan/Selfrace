<p align="center">
  <a href="https://github.com/HongfeiWan/Selfrace">
    <img src="./images/Logo.png" width="1000" alt="Selfrace">
  </a>
</p>

<p align="center">
  <a href="https://github.com/pytorch/pytorch">
    <img src="https://img.shields.io/badge/pytorch-2.8.0-brightgreen.svg" alt="PyTorch 2.8.0">
  </a>
  <a href="https://github.com/NVIDIA/cuda-python">
    <img src="https://img.shields.io/badge/cudapython-13.0.1-brightgreen.svg" alt="CUDA Python 13.0.1">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-GPLv3-blue.svg" alt="GPLv3">
  </a>
</p>

<p align="center">
  <a href="./README.md">简体中文</a> · <strong>English</strong>
</p>

# Selfrace

Selfrace is a GPU-parallel multi-agent simulator and self-play PPO training project for autonomous-driving research. It uses preprocessed road maps to simulate many traffic worlds and vehicles in PyTorch while training one shared driving policy.

The repository is informed by the environment, observation, action, randomization, and advantage-filtered PPO design in [Robust Autonomy Emerges from Self-Play](https://arxiv.org/abs/2502.03349). It is an independent research implementation, not the authors' official code, and does not claim result-by-result equivalence with the paper.

> [!WARNING]
> Selfrace is research software, not a safety-certified autonomous-driving system. Do not use it to control real vehicles.

## Highlights

- **Batched GPU simulation:** one process maintains many worlds, each with multiple learning vehicles.
- **One training entry point:** single-GPU and DDP runs share the same rollout/PPO loop. Select devices with `--gpus`; no manual `torchrun` command is required.
- **VRAM adaptation:** startup scales the world count to free VRAM; rollout, feature caching, and PPO microbatches back off on CUDA OOM and recover after stable updates.
- **Self-play driving task:** one policy controls every active vehicle, while vehicle dimensions, dynamics style, reward coefficients, and route targets are sampled independently.
- **Structured observations:** state, route goals, reward conditions, road boundaries, lane points, stop lines, and nearby agents share one named feature schema.
- **Independent actor and critic:** policy and value networks do not share weights; set inputs use permutation-invariant encoders.
- **Complete training resume:** current checkpoints include the model, optimizers, schedulers, AMP state, RNG state, and adaptive-memory state.
- **SwanLab tracking:** rank 0 can report rewards, event rates, PPO metrics, timings, and CUDA memory.
- **Windows inference UI:** a Pygame top-down view shows vehicles, roads, routes, all 12 action probabilities, and the exact roadside points observed by the selected car.

## Requirements

| Use case | Requirement |
| --- | --- |
| Training | NVIDIA GPU and a CUDA-enabled PyTorch build; Linux is recommended for multi-GPU runs |
| Windows viewer | Python 3.10+; CPU and CUDA are supported; a VS Code F5 profile is included |
| Multi-GPU | One process per GPU; NCCL on Linux and Gloo on Windows |
| Optional tools | SwanLab, interactive Matplotlib diagnostics, and pytest |

The badges above preserve the project's original reference-environment markers. The supported package ranges in [`requirements.txt`](./requirements.txt) are authoritative. Before training, verify that PyTorch can see CUDA:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

## Installation

```powershell
git clone https://github.com/HongfeiWan/Selfrace.git
cd Selfrace

# Choose one: uv can fetch Python 3.12; use the commented command when Python 3.12 is installed system-wide
uv venv --python 3.12 --seed .venv
# py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux, create and activate the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For training, first install the CUDA build matching your driver from the [official PyTorch installation page](https://pytorch.org/get-started/locally/), then install the remaining requirements. Run commands from the repository root so relative map and configuration paths resolve correctly.

## Quick inference demo

The repository includes a viewer-only checkpoint at `training/checkpoints-paper-goals/ckpt_step_400.pt`.

### Press F5 in VS Code

1. Open the repository root in VS Code.
2. Install Microsoft's Python and Python Debugger extensions.
3. Create `.venv` and install the requirements as described above.
4. Select **Selfrace Game Inference** and press `F5`.

The committed [`.vscode/launch.json`](./.vscode/launch.json) selects `.venv`, the example checkpoint, CPU inference, and 24 vehicles.

### Start from a terminal

```powershell
python .\game\game.py `
  --checkpoint .\training\checkpoints-paper-goals\ckpt_step_400.pt `
  --device cpu `
  --envs 1 `
  --agents 24
```

Change `--device cpu` to `--device cuda:0` for CUDA inference. For a windowless check:

```powershell
python .\game\game.py `
  --checkpoint .\training\checkpoints-paper-goals\ckpt_step_400.pt `
  --device cpu `
  --envs 1 `
  --agents 24 `
  --headless `
  --steps 120
```

The bundled checkpoint uses an earlier weights-only format and is intended for `game/game.py`. The current trainer's `--resume` option only accepts complete, versioned checkpoints produced by the current trainer.

### Viewer guide

- Green vehicle: the currently observed vehicle.
- Gold vehicles: other active vehicles.
- Bright cyan dots: the exact `W_boundary` roadside points supplied to the selected vehicle's network observation.
- Right-hand action panel: all 12 discrete actions and their current probabilities.

| Key | Action |
| --- | --- |
| `Space` | Pause/resume |
| `Tab` | Select the next active vehicle |
| `[` / `]` | Switch world |
| `+` / `-` | Zoom |
| `B` | Toggle observed road-boundary points |
| `C` | Toggle north-up/ego-aligned camera |
| `W` | Toggle other vehicles' route points |
| `M` | Toggle argmax/action sampling |
| `R` | Reset |
| `Esc` | Exit |

## Training

### Small smoke run

The default configuration targets a large, long-running job. Start by overriding the number of worlds per GPU:

```powershell
python .\training\train.py `
  --gpus 0 `
  --num-envs 8 `
  --updates 2 `
  --checkpoint-dir .\training\checkpoints-smoke
```

The training entry point requires CUDA and exits if no GPU is available.

### Single-GPU training

```powershell
python .\training\train.py `
  --gpus 0 `
  --updates 10000 `
  --checkpoint-dir .\training\checkpoints
```

### Multi-GPU DDP training

```bash
python training/train.py \
  --gpus 0,1,2,3 \
  --updates 10000 \
  --checkpoint-dir training/checkpoints-ddp
```

`simulator.num_envs` is the world count **per GPU**, so the global world count is `num_envs × number of GPUs`. The launcher starts workers and chooses a local rendezvous port itself.

### Resume a run

```powershell
python .\training\train.py `
  --resume .\training\checkpoints\latest.pt `
  --updates 20000 `
  --gpus 0 `
  --checkpoint-dir .\training\checkpoints
```

`--updates` is the **final update number**, not an additional-update count. One update always consists of one rollout followed by one PPO update. Current checkpoints are named `ckpt_update_<N>.pt`, and `latest.pt` is replaced atomically in the same directory.

### SwanLab

After logging into SwanLab on the machine, enable tracking from the CLI:

```bash
python training/train.py \
  --gpus 0 \
  --updates 10000 \
  --checkpoint-dir training/checkpoints-swanlab \
  --swanlab \
  --swanlab-project Selfrace \
  --swanlab-experiment selfrace-longrun \
  --swanlab-id selfrace-longrun-01 \
  --swanlab-resume allow
```

Only rank 0 creates and uploads an experiment. Keep credentials in SwanLab's user-level configuration; never commit them to this repository.

## Core design

### Simulation loop

`TeraflowSimulator` coordinates these stages:

1. update the kinematic bicycle model from one of 12 discrete jerk actions;
2. check continuous vehicle collisions, static collisions, and off-road events;
3. compute Frenet coordinates, stop-line events, and rewards;
4. advance route targets and apply masked world resets at rollout boundaries;
5. rebuild network features on demand for active vehicles.

Maps are preprocessed into road quadrilaterals, centerlines, boundaries, routing data, and traffic controls. Training and inference do not require a running CARLA instance.

### Action space

Actions are the Cartesian product of longitudinal and lateral jerk:

- longitudinal jerk: `{-15, -4, 0, 4} m/s³`
- lateral jerk: `{-4, 0, 4} m/s³`
- total: `4 × 3 = 12` actions

### Observations and networks

The default named schema has an input width of 977:

| Group | Width | Meaning |
| --- | ---: | --- |
| `state` | 13 | Ego-local state and dynamics state |
| `goal` | 8 | Relative positions of up to four remaining route goals |
| `reward` | 12 | Per-vehicle reward-conditioning parameters |
| `vehicle_style` | 4 | Vehicle dynamics/driving-style parameters |
| `road_boundary` | 160 | Road-boundary point set |
| `lane_points` | 560 | Lane geometry and route-distance set |
| `stop_lines` | 20 | Stop-line set |
| `other_agents` | 200 | Nearby-agent set with validity markers |

The feature builder, actor, and critic use the same layout validation from [`training/feature_schema.py`](./training/feature_schema.py). Actor and critic parameters remain independent and run sequentially to control peak VRAM.

### Routes, randomization, and rewards

- Each active vehicle first samples a goal uniformly from routable road quads, followed by 0–3 additional goals.
- Later goals first target a 20–200 m distance and at most a 60° lane-heading change; constraints are relaxed progressively when no candidate is available.
- Worlds randomize vehicle dimensions, driving style, reward weights, and traffic-light state.
- Reward terms cover goals, collisions, off-road events, comfort, lane heading, lane centering, velocity, reversing, stop lines, and timestep cost.

### PPO and memory policy

- Rollout builds features only for active vehicles; full observations are not stored in the rollout buffer.
- Training uses GAE, PPO clipping, advantage filtering, and multiple optimization epochs.
- Selected PPO features are built once per update and can move to pinned CPU memory when GPU headroom is insufficient.
- Microbatch accumulation preserves the effective PPO sample target instead of silently shrinking it after an OOM.
- Collision detection uses bounded blocks controlled by `collision_stream_pair_budget` to cap pair-shaped temporary tensors.
- Linux CUDA can use controlled `torch.compile` for network and collision blocks; Windows, CPU, small batches, or compilation failures fall back to eager execution.

## Key configuration

All runtime settings live in [`configs/default_config.yaml`](./configs/default_config.yaml). Important defaults are:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `simulator.num_envs` | 2600 | Parallel worlds per GPU |
| `simulator.max_agents_num` | 150 | Vehicle-slot limit per world |
| `simulator.sim_dt` | 0.3 | Simulation timestep in seconds |
| `training.rollout_length` | 128 | Rollout steps before each PPO update |
| `training.batch_size_per_gpu` | 32000 | PPO sample target per rank |
| `training.ppo_epochs` | 3 | PPO epochs over each selected batch |
| `training.gamma` | 0.999 | Discount factor |
| `training.gae_lambda` | 0.95 | GAE coefficient |
| `training.learning_rate` | 5e-4 | Actor and critic learning rate |
| `training.precision` | `16-bit` | CUDA AMP precision |
| `training.total_updates` | 10000 | Final update number |

Important constraints:

- `max_episode_length` must be divisible by `rollout_length`, because worlds reset at rollout boundaries.
- The default `2600 × 150` target is not an appropriate first smoke test.
- `memory_adaptation` reduces the world count from startup free VRAM and records safe rollout, feature-building, and PPO chunk sizes independently.
- Network input requires the named `feature_schema`; legacy parallel dimension lists are intentionally unsupported.
- The current training checkpoint format has no legacy migration path, preventing silent loss of optimizer or adaptive-memory state.

## Repository layout

```text
Selfrace/
├── configs/          # Simulator, PPO, network, and logging defaults
├── simulator/        # Initialization, dynamics, roads, observations, collision, off-road, rewards
├── training/         # Single/DDP loop, networks, feature schema, checkpoints, SwanLab
├── game/             # Pygame checkpoint-inference viewer
├── maps/             # CARLA OpenDRIVE, processed maps, and map build/inspection scripts
├── tests/            # Deterministic unit and structural tests
├── scripts/manual/   # Interactive diagnostics requiring a window, CUDA, or large maps
├── paper/            # English and Chinese technical manuscripts stored with the project
├── images/           # Logo and project images
└── course/           # Network experimentation notebook
```

The default map is `maps/processed_map_Town01_stitched.json`. Several additional Town maps and `.xodr` sources are included. When changing `simulator.map_path`, ensure that the corresponding routing and junction-processing data exists.

## Tests and manual diagnostics

After installing pytest, run:

```powershell
python -m pytest -q
```

Interactive plots and larger component checks are under [`scripts/manual/`](./scripts/manual/README.md):

```powershell
python .\scripts\manual\network_demo.py
python .\scripts\manual\simulator_demo.py
```

Some scripts require CUDA, map assets, or an interactive Matplotlib window. Fast deterministic regression checks belong in `tests/`.

## Resources

- Reference paper: [Robust Autonomy Emerges from Self-Play](https://arxiv.org/abs/2502.03349)
- Included English manuscript: [paper/main.pdf](./paper/main.pdf)
- Included Chinese manuscript: [paper/selfplay_selfrace_zh.pdf](./paper/selfplay_selfrace_zh.pdf)
- Manual diagnostics: [scripts/manual/README.md](./scripts/manual/README.md)

## License

Selfrace is licensed under the [GNU General Public License v3.0](./LICENSE).
