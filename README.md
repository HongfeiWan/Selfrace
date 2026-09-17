<p align="center">
    <a href="https://github.com/HongfeiWan/Selfrace" target="_blank">
        <img src="https://github.com/HongfeiWan/Selfrace/tree/main/images/Logo.png" width="1000">
    </a>
</p>
<p align="center">
    <a href="https://github.com/pytorch/pytorch">
        <img src="https://img.shields.io/badge/pytorch-2.8.0-brightgreen.svg">
    </a>
    <a href="https://github.com/NVIDIA/cuda-python">
        <img src="https://img.shields.io/badge/cudapython-13.0.1-brightgreen.svg">
    </a>
</p>

Selfrace is a simulator for self-play end-to-end auto-driving training.

## Installation

```bash
# Clone the repository
git clone https://github.com/HongfeiWan/Selfrace.git
cd Selfrace

# Install dependencies
pip install -r requirements.txt
```

## Documentation

For detailed documentation, tutorials, and API references, please visit our [Wiki](https://github.com/HongfeiWan/Selfrace/wiki).

The wiki contains comprehensive information about:

- Core modules and their usage
- Configuration guides
- API references
- Performance optimization
- Troubleshooting guides
- And much more!

## Quick Start

Run training from the repository root. The same worker and rollout/PPO loop is
used for one GPU and DDP; all CUDA devices visible through
`CUDA_VISIBLE_DEVICES` are selected automatically.

```powershell
python .\training\train.py
```

Resume a checkpoint or select a configuration/GPU without editing production
code. One update means exactly one rollout followed by one PPO update;
`--updates` is the final update number. Checkpoints are named
`ckpt_update_<N>.pt` and restore optimizer, scheduler, AMP, per-rank RNG, and
adaptive-memory state. If `--updates` changes on resume, the loaded cosine
scheduler is reprojected to that new final update instead of retaining its old
hidden `T_max`.

```powershell
python .\training\train.py --resume C:\checkpoints\latest.pt --updates 10000 --gpus 0
```

For a restartable tracked run, add `--swanlab`, a stable `--swanlab-id`, and
`--swanlab-resume allow`; `--checkpoint-dir` keeps that run's snapshots
isolated. These command-line overrides are included in the configuration sent
to SwanLab.

## Training configuration

`configs/default_config.yaml` follows the PPO settings in the paper where they
map directly to this implementation:

| Configuration | Default | Runtime meaning |
| --- | ---: | --- |
| `batch_size` | 256,000 | Paper's global target; also the fallback used to derive a per-rank target when `batch_size_per_gpu` is absent. |
| `batch_size_per_gpu` | 32,000 | Target ceiling for PPO samples selected by each rank. If fewer eligible samples exist, the available count is used. |
| `total_updates` | 10,000 | Final `update_step`; every step is one rollout followed by one PPO call. |
| `ppo_microbatch_initial` / `ppo_microbatch_max` | `null` / `null` | CUDA-memory-adaptive gradient-accumulation chunk. `null` starts at, and never exceeds, the effective batch size. |
| `ppo_min_microbatch_size` / `ppo_microbatch_growth_interval` | 256 / 20 | Retry floor and number of stable updates before attempting a larger microbatch. |
| `rollout_length` / `ppo_epochs` | 128 / 3 | On-policy rollout horizon and optimization epochs. |
| `gamma` / `gae_lambda` | 0.999 / 0.95 | Return discount and GAE coefficient. |
| `max_episode_length` | 1,280 | World time limit. It must be divisible by `rollout_length` because masked world resets happen at rollout boundaries. |
| `clip_ratio` / `value_clip_ratio` | 0.2 / `null` | Policy clipping; `null` means no value-function clipping, as in the paper. A numeric value enables it. |
| `learning_rate` / `lr_schedule` | 5e-4 / `cosine` | Adam learning rate and the scheduler actually constructed by the trainer. |
| `entropy_coef` / `value_loss_coef` / `max_grad_norm` | 0.01 / 0.5 / 0.5 | PPO loss coefficients and gradient clipping. |
| `advantage_filter_threshold` / `advantage_filter_beta` | 0.01 / 0.25 | Advantage threshold and EWMA coefficient. |
| `precision` | `16-bit` | CUDA automatic mixed precision. |
| `network.compile` | enabled (`default`, dynamic) | Controlled actor/critic compilation on Linux CUDA with Triton. CPU, Windows, small chunks, and a failed backend automatically use eager execution. |
| `w_lane_dropout_prob` / `w_boundary_dropout_prob` | 0.5 / 0.4 | Element dropout applied while observations are rebuilt from world state. |
| `weight_init` | orthogonal, zero bias | Initialization applied to the actor and critic linear layers. |

`advantage_filter_max_drop_fraction` and `min_ppo_samples` are active
implementation safety guards. `memory_adaptation`, the rollout/feature/PPO
chunk controls, `profile`, and `diagnostics` are implementation controls rather
than paper hyperparameters. Diagnostics are off by default.

Route targets follow the paper's Appendix A.1/B.4 setup. Every active vehicle
independently receives a first target sampled uniformly from routable map
quads, then receives 0--3 additional targets. Each additional target first
tries the paper's 20--200 m distance window and at most 60 degree lane-heading
change; the constraints are progressively relaxed when no reachable candidate
exists. The former short first-target and 15--80 m curriculum settings were
removed because they do not describe the paper's sampling distribution.

`batch_size_per_gpu` controls the effective optimization sample target, not one
indivisible CUDA allocation. Selected PPO features are built once per update
and reused for all epochs; the cache moves to pinned CPU when GPU headroom is
insufficient. Rollout feature construction is streamed in adaptive chunks, and
PPO accumulates sample-weighted gradients without shrinking the effective
batch. At startup, free VRAM caps `num_envs` and the complete rollout storage is
preallocated; a failed preflight halves the world count and retries. Safe chunk
sizes are remembered and checkpointed. DDP reduces flattened gradient buckets
after every rank succeeds instead of issuing one collective per parameter.
Checkpoint format v3 intentionally has no legacy migration path; incompatible
checkpoints fail before training rather than silently dropping optimizer state.

The rollout path compacts alive vehicles and builds only their ego features.
Within each packed chunk, sanitized features, schema slices, set-validity masks,
and compact set indices are shared by the independent actor and critic; the
actor graph is released before the critic runs. Dynamic vehicle collision checking streams a
bounded number of unordered pairs per block. `collision_stream_pair_budget`
is a hard upper bound for a block's pair-shaped temporary tensors, and the
compiled Linux/CUDA path keeps candidate counts on-device instead of reading
them into Python.

SwanLab tracking is optional and initialized only by rank 0. Set
`training.swanlab.enabled: true` and provide a project/experiment name in the
YAML; update reward/event rates, PPO losses and KL, adaptive microbatch/OOM
statistics, learning rates, timings, and peak CUDA memory are then logged. A
fixed `id` with `resume: "allow"` lets a restarted process append to the same
SwanLab experiment. Login credentials remain in SwanLab's user-level config
and must not be placed in this repository.

The paper reports roughly `1e12` sampled transitions as a training scale, not as
an algorithm parameter. This repository terminates by `training.total_updates`, so
the former unused `total_timesteps` key was removed. The old replay-buffer size
keys were also removed: PPO uses a fixed-horizon on-policy `RolloutTensorBuffer`,
not an experience replay buffer. DDP selects `gloo` on Windows and `nccl` on
non-Windows CUDA at runtime and allocates a free local rendezvous port, so the
unused backend/init keys are no longer exposed.

Two nominal numerical knobs, `curvature_epsilon` and
`straight_motion_threshold`, were assigned but never used by the dynamics and
have no corresponding paper hyperparameter, so they were removed.
`steering_epsilon` remains because the bicycle-model update uses it directly.
The stale fixed `simulator.device`, `vehicle_length`, and `vehicle_width` values
were also removed: each worker's actual device is authoritative, while vehicle
dimensions are sampled from the configured min/max ranges.
The fixed wheelbase/control-coefficient defaults were removed from the YAML
because training supplies per-vehicle `0.6 * length` wheelbases and sampled
`C_dynamics` values; internal neutral fallbacks remain for standalone API use.
The unused `hash.grid_width`, `grid_height`, and `map_extent_m` values were
removed as well; the real grid extents come from the loaded map bounds and only
`hash_cell_size` controls its resolution.

The action space remains the same 12-action Cartesian product defined by
`DiscreteActionSpace`: longitudinal jerk `{-15, -4, 0, 4}` and lateral jerk
`{-4, 0, 4}`. `training.network.num_actions` is checked against the simulator at
startup; redundant action enable/size flags were removed.

Network input layout now has one named `training.network.feature_schema`, shared
by feature construction and both independent actor/critic encoders. Its groups
are `state`, `goal`, `reward`, `vehicle_style`, `road_boundary`, `lane_points`,
`stop_lines`, and `other_agents`; startup validation rejects dimensions that do
not match the simulator observation settings. The named schema is required;
legacy parallel dimension lists are not accepted. Actor and critic remain fully
independent, so the unused shared-network implementation was removed.

Interactive plots and ad-hoc module checks live under `scripts/manual/` rather
than production modules. See [manual diagnostics](scripts/manual/README.md).

## Windows inference viewer

Create the environment with a Windows Python path, install the project dependencies,
then point the viewer at a trained checkpoint:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\game\game.py --checkpoint .\training\checkpoints_remote\latest.pt
```

The green vehicle is the currently observed car; other active vehicles are gold.
Bright cyan dots are the exact `W_boundary` roadside points used by that car's
network observation. The right panel keeps all 12 existing longitudinal/lateral
jerk actions and shows their current probabilities.

Viewer controls: `Space` pause, `Tab` next car, `[`/`]` switch world, `+`/`-`
zoom, `B` toggle observed roadside points, `C` toggle ego-aligned/north-up camera,
`W` toggle other vehicles' route points, `M` switch argmax/sampling, and `R` reset.

## License

This project is licensed under the GNU General Public License v3.0 License - see the [LICENSE](LICENSE) file for details.

## References

- [Robust Autonomy Emerges from Self-Play](https://arxiv.org/abs/2502.03349)
