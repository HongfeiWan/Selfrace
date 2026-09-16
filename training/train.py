"""Command-line entry point for single- and multi-GPU PPO training."""

import argparse
import sys
from pathlib import Path

import yaml

from ddppo import check_gpu_info, run_distributed_ddppo


def parse_args():
    default_config = Path(__file__).resolve().parents[1] / "configs" / "default_config.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config, help="YAML configuration path")
    parser.add_argument("--resume", type=Path, help="Checkpoint used to initialize/resume training")
    parser.add_argument("--updates", type=int, help="Total PPO update at which training stops")
    parser.add_argument("--num-envs", type=int, help="Override parallel environments per GPU")
    parser.add_argument("--checkpoint-dir", type=Path, help="Directory for periodic and latest checkpoints")
    parser.add_argument("--swanlab", action="store_true", help="Enable rank-zero SwanLab logging")
    parser.add_argument("--swanlab-project", help="SwanLab project name")
    parser.add_argument("--swanlab-experiment", help="SwanLab experiment name")
    parser.add_argument("--swanlab-id", help="Stable SwanLab run ID used for resumption")
    parser.add_argument(
        "--swanlab-resume",
        choices=("allow", "must", "never"),
        help="SwanLab experiment resumption policy",
    )
    parser.add_argument(
        "--gpus",
        help="Comma-separated visible CUDA device indices; defaults to every visible device",
    )
    return parser.parse_args()


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
        except Exception:
            pass
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if args.resume is not None:
        config["training"]["resume_from"] = str(args.resume.resolve())
    if args.updates is not None:
        if args.updates <= 0:
            raise SystemExit("--updates must be positive")
        config["training"]["total_updates"] = args.updates
    if args.num_envs is not None:
        if args.num_envs <= 0:
            raise SystemExit("--num-envs must be positive")
        config["simulator"]["num_envs"] = args.num_envs
    if args.checkpoint_dir is not None:
        config["training"]["checkpoint_dir"] = str(args.checkpoint_dir.resolve())

    swanlab_cfg = config["training"].setdefault("swanlab", {})
    if args.swanlab:
        swanlab_cfg["enabled"] = True
    for argument, key in (
        (args.swanlab_project, "project"),
        (args.swanlab_experiment, "experiment_name"),
        (args.swanlab_id, "id"),
        (args.swanlab_resume, "resume"),
    ):
        if argument is not None:
            swanlab_cfg["enabled"] = True
            swanlab_cfg[key] = argument

    cuda_available, ranks = check_gpu_info(print_info=False)
    if args.gpus:
        try:
            requested_ranks = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
        except ValueError as exc:
            raise SystemExit("--gpus must be a comma-separated list of integer indices") from exc
        invalid = [rank for rank in requested_ranks if rank not in ranks]
        if not requested_ranks or invalid:
            raise SystemExit(f"invalid CUDA device selection {requested_ranks}; available indices: {ranks}")
        ranks = requested_ranks
    print(f"CUDA available: {cuda_available}, ranks: {ranks}")
    if not cuda_available or not ranks:
        raise SystemExit("No CUDA device is available")
    run_distributed_ddppo(config, ranks)


if __name__ == "__main__":
    main()








