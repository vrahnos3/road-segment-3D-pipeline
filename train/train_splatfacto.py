"""Train a Splatfacto 3D Gaussian Splatting scene from a processed Nerfstudio dataset.

This script is intentionally thin: it reuses Nerfstudio's built-in Splatfacto
TrainerConfig and only overrides the parts that are useful for automation.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Literal, Optional, cast

import torch

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_config import publish_training_config

from nerfstudio.configs.method_configs import method_configs
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.scripts.train import main as train_main
from nerfstudio.models.splatfacto import SplatfactoModelConfig


VisOption = Literal[
    "viewer",
    "wandb",
    "tensorboard",
    "comet",
    "viewer+wandb",
    "viewer+tensorboard",
    "viewer+comet",
    "viewer_legacy",
]


def _validate_dataset(dataset_dir: Path) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path must be a directory: {dataset_dir}")
    if not (dataset_dir / "transforms.json").exists():
        raise FileNotFoundError(
            f"Expected a Nerfstudio dataset with transforms.json at: {dataset_dir / 'transforms.json'}"
        )


def build_splatfacto_config(
    dataset_dir: Path,
    output_dir: Path,
    *,
    experiment_name: Optional[str] = None,
    max_num_iterations: int = 30000,
    downscale_factor: Optional[int] = None,
    vis: VisOption = "viewer",
    cache_images: Literal["cpu", "gpu", "disk"] = "gpu",
    cache_images_type: Literal["uint8", "float32"] = "uint8",
    steps_per_save: int = 2000,
    steps_per_eval_image: int = 100,
    steps_per_eval_all_images: int = 1000,
    device_type: Literal["cuda", "cpu", "mps"] = "cuda",
    num_devices: int = 1,
    quit_on_train_completion: bool = True,
) -> TrainerConfig:
    """Create a Splatfacto trainer config using Nerfstudio's canonical defaults."""

    _validate_dataset(dataset_dir)

    config = deepcopy(method_configs["splatfacto"])
    if not isinstance(config, TrainerConfig):
        raise TypeError("method_configs['splatfacto'] did not resolve to a TrainerConfig.")

    config.output_dir = output_dir
    config.output_dir_is_run_dir = True
    config.experiment_name = experiment_name or dataset_dir.name
    config.data = dataset_dir
    config.max_num_iterations = max_num_iterations
    config.steps_per_save = steps_per_save
    config.steps_per_eval_image = steps_per_eval_image
    config.steps_per_eval_all_images = steps_per_eval_all_images
    config.vis = vis
    config.machine.device_type = device_type
    config.machine.num_devices = num_devices
    config.viewer.quit_on_train_completion = quit_on_train_completion

    datamanager = cast(FullImageDatamanagerConfig, config.pipeline.datamanager)
    datamanager.data = dataset_dir
    datamanager.cache_images = cache_images
    datamanager.cache_images_type = cache_images_type

    dataparser = cast(NerfstudioDataParserConfig, datamanager.dataparser)
    dataparser.data = dataset_dir
    dataparser.downscale_factor = downscale_factor
    dataparser.load_3D_points = True

    """
    Optional args for possible improved results regarding the long and noisy gaussians
    """
    model = cast(SplatfactoModelConfig, config.pipeline.model)

    model.use_scale_regularization = True
    model.max_gauss_ratio = 5.0
    model.cull_scale_thresh = 0.05
    model.cull_screen_size = 0.05

    return config


def train_splatfacto(
    dataset_dir: Path,
    output_dir: Path,
    *,
    experiment_name: Optional[str] = None,
    max_num_iterations: int = 30000,
    downscale_factor: Optional[int] = None,
    vis: VisOption = "viewer",
    cache_images: Literal["cpu", "gpu", "disk"] = "gpu",
    cache_images_type: Literal["uint8", "float32"] = "uint8",
    steps_per_save: int = 2000,
    steps_per_eval_image: int = 100,
    steps_per_eval_all_images: int = 1000,
    device_type: Literal["cuda", "cpu", "mps"] = "cuda",
    num_devices: int = 1,
    quit_on_train_completion: bool = True,
    print_to_terminal: bool = False,
    training_config_path: Optional[Path] = None,
) -> TrainerConfig:
    """Build the config, optionally print it, and run training."""

    torch.set_float32_matmul_precision("high")
    config = build_splatfacto_config(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        max_num_iterations=max_num_iterations,
        downscale_factor=downscale_factor,
        vis=vis,
        cache_images=cache_images,
        cache_images_type=cache_images_type,
        steps_per_save=steps_per_save,
        steps_per_eval_image=steps_per_eval_image,
        steps_per_eval_all_images=steps_per_eval_all_images,
        device_type=device_type,
        num_devices=num_devices,
        quit_on_train_completion=quit_on_train_completion,
    )

    if print_to_terminal:
        config.print_to_terminal()
        return config

    # A fixed run directory must not mix old checkpoints with a new run, or an
    # interrupted run could appear complete because an old final checkpoint exists.
    if any(config.get_checkpoint_dir().glob("*.ckpt")):
        raise FileExistsError(
            f"Training output already contains checkpoints: {config.get_checkpoint_dir()}. "
            "Choose a different --output_dir (pipeline: --splatfacto_output_dir), "
            "or move the previous training output before starting a new run."
        )
    train_main(config)
    publish_training_config(
        config.get_base_dir() / "config.yml",
        training_config_path or output_dir / "config.yml",
        config.get_checkpoint_dir(),
        config.max_num_iterations - 1,
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Splatfacto scene from a processed Nerfstudio dataset.")
    parser.add_argument(
        "-d", "--dataset_dir", required=True, type=Path, help="Dataset folder containing transforms.json."
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=Path, help="Directory for config, logs, and checkpoints."
    )
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name. Defaults to dataset folder name.")
    parser.add_argument("--training_config_path", type=Path,
                        help="Stable config path for export (default: output_dir/config.yml).")
    parser.add_argument("--max_num_iterations", type=int, default=30000, help="Total training iterations.")
    parser.add_argument(
        "--downscale_factor", type=int, choices=[1, 2, 4, 8], default=None, help="Image downscale factor."
    )
    parser.add_argument(
        "--vis",
        choices=[
            "viewer",
            "wandb",
            "tensorboard",
            "comet",
            "viewer+wandb",
            "viewer+tensorboard",
            "viewer+comet",
            "viewer_legacy",
        ],
        default="viewer",
        help="Nerfstudio visualizer backend.",
    )
    parser.add_argument("--cache_images", choices=["cpu", "gpu", "disk"], default="gpu", help="Where to cache images.")
    parser.add_argument("--cache_images_type", choices=["uint8", "float32"], default="uint8", help="Cached image dtype.")
    parser.add_argument("--steps_per_save", type=int, default=2000, help="Checkpoint save interval.")
    parser.add_argument("--steps_per_eval_image", type=int, default=100, help="Single-image eval interval.")
    parser.add_argument("--steps_per_eval_all_images", type=int, default=1000, help="Full eval interval.")
    parser.add_argument("--device_type", choices=["cuda", "cpu", "mps"], default="cuda", help="Training device type.")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of devices for training.")
    parser.add_argument(
        "--keep_viewer_open",
        action="store_true",
        help="Keep the viewer process alive after training finishes. This prevents the timer from printing until Ctrl+C.",
    )
    parser.add_argument("--print_to_terminal", action="store_true", help="Print the final config without training.")
    args = parser.parse_args()

    start_time = perf_counter()
    train_splatfacto(
        dataset_dir=args.dataset_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        experiment_name=args.experiment_name,
        max_num_iterations=args.max_num_iterations,
        downscale_factor=args.downscale_factor,
        vis=cast(VisOption, args.vis),
        cache_images=args.cache_images,
        cache_images_type=args.cache_images_type,
        steps_per_save=args.steps_per_save,
        steps_per_eval_image=args.steps_per_eval_image,
        steps_per_eval_all_images=args.steps_per_eval_all_images,
        device_type=args.device_type,
        num_devices=args.num_devices,
        quit_on_train_completion=not args.keep_viewer_open,
        print_to_terminal=args.print_to_terminal,
        training_config_path=args.training_config_path.expanduser().resolve() if args.training_config_path else None,
    )
    end_time = perf_counter()
    total_time = end_time - start_time
    formatted_time = str(timedelta(seconds=total_time))
    print(f"Execution time: {formatted_time}")


if __name__ == "__main__":
    main()
