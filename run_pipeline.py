"""Run the complete drone-video-to-Splatfacto pipeline from one YAML config."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Optional


PIPELINE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PIPELINE_ROOT.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PipelineConfigError,
    PipelinePaths,
    get_config_section,
    load_pipeline_config,
)


def run_stage(name: str, command: list[str]) -> None:
    """Run one pipeline stage and stop the pipeline if it fails."""

    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}", flush=True)
    print(f"$ {shlex.join(command)}", flush=True)
    start_time = perf_counter()
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    elapsed = perf_counter() - start_time
    print(f"{name} completed in {elapsed:.2f} seconds.", flush=True)


def _setting(cli_value: Any, section: Mapping[str, Any], name: str, default: Any) -> Any:
    return cli_value if cli_value is not None else section.get(name, default)


def check_reconstruction(
    transforms_path: Path, image_dir: Path, min_images: int, min_fraction: float
) -> None:
    """Reject insufficient registration before starting training."""
    if isinstance(min_images, bool) or not isinstance(min_images, int) or min_images < 1:
        raise PipelineConfigError("reconstruction.min_registered_images must be a positive integer")
    if isinstance(min_fraction, bool) or not isinstance(min_fraction, (int, float)) or not 0 <= min_fraction <= 1:
        raise PipelineConfigError("reconstruction.min_registered_fraction must be between 0 and 1")
    metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = metadata.get("frames") if isinstance(metadata, dict) else None
    if not isinstance(frames, list):
        raise RuntimeError(f"Invalid reconstruction: {transforms_path} has no frames list")
    registered = len(frames)
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    total = sum(p.is_file() and p.suffix.lower() in extensions for p in image_dir.iterdir())
    fraction = registered / total if total else 0
    print(f"Reconstruction: {registered}/{total} images registered ({fraction:.1%}).", flush=True)
    if not total or registered < min_images or fraction < min_fraction:
        raise RuntimeError(
            f"Insufficient reconstruction: {registered}/{total} images registered ({fraction:.1%}); "
            f"requires at least {min_images} images and {min_fraction:.0%} registration. "
            "Training, export, and the Kafka ready notification were not started. "
            "Review reconstruction settings or explicitly adjust reconstruction.min_registered_images "
            "and reconstruction.min_registered_fraction."
        )


def _add_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video_path", type=Path, help="Override paths.video_path.")
    parser.add_argument("--dataset_dir", type=Path, help="Override the root folder for generated data.")
    parser.add_argument("--image_dir", type=Path, help="Override the extracted image folder.")
    parser.add_argument("--colmap_dir", type=Path, help="Override the hloc/COLMAP output folder.")
    parser.add_argument(
        "--nerfstudio_dir",
        type=Path,
        help="Override the folder containing transforms.json and sparse_pc.ply.",
    )
    parser.add_argument("--splatfacto_output_dir", type=Path, help="Override the training output folder.")
    parser.add_argument("--training_config_path", type=Path, help="Override the generated training config path.")
    parser.add_argument("--export_dir", type=Path, help="Override the model export folder.")
    parser.add_argument("--exported_ply_path", type=Path, help="Override the exported PLY path.")
    parser.add_argument("--splat_output_path", type=Path, help="Override the converted SPLAT path.")
    parser.add_argument("--kafka_message_path", type=Path, help="Override the Kafka JSON template path.")
    parser.add_argument(
        "--converter_executable",
        help="Override the 3dgsconverter executable path or command name.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run extraction, reconstruction, Splatfacto training, export, and publishing."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Pipeline YAML file (default: {DEFAULT_CONFIG_PATH}).",
    )
    _add_path_arguments(parser)

    parser.add_argument("--num_frames_target", type=int)
    parser.add_argument("--num_downscales", type=int)
    parser.add_argument("--num_matched", type=int)
    parser.add_argument("--min_registered_images", type=int)
    parser.add_argument("--min_registered_fraction", type=float)
    parser.add_argument("--train_downscale_factor", type=int, choices=[1, 2, 4, 8])
    parser.add_argument("--max_num_iterations", type=int)
    parser.add_argument("--splat_compression_level", type=int)
    parser.add_argument("--splat_min_opacity", type=int)
    parser.add_argument("--obb_center", type=float, nargs=3)
    parser.add_argument("--obb_rotation", type=float, nargs=3)
    parser.add_argument("--obb_scale", type=float, nargs=3)
    parser.add_argument("--viewer_port", type=int)

    extraction_group = parser.add_mutually_exclusive_group()
    extraction_group.add_argument(
        "--skip_image_extraction", dest="skip_image_extraction", action="store_true"
    )
    extraction_group.add_argument(
        "--extract_images", dest="skip_image_extraction", action="store_false"
    )
    kafka_group = parser.add_mutually_exclusive_group()
    kafka_group.add_argument("--send_kafka_message", dest="send_kafka_message", action="store_true")
    kafka_group.add_argument("--no_send_kafka_message", dest="send_kafka_message", action="store_false")
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument("--start_viewer", dest="start_viewer", action="store_true")
    viewer_group.add_argument("--no_start_viewer", dest="start_viewer", action="store_false")
    parser.set_defaults(skip_image_extraction=None, send_kafka_message=None, start_viewer=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve and print all paths without creating files or running a stage.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_pipeline_config(config_path)
    path_overrides = {
        name: getattr(args, name)
        for name in (
            "video_path",
            "dataset_dir",
            "image_dir",
            "colmap_dir",
            "nerfstudio_dir",
            "splatfacto_output_dir",
            "training_config_path",
            "export_dir",
            "exported_ply_path",
            "splat_output_path",
            "kafka_message_path",
            "converter_executable",
        )
    }
    paths = PipelinePaths.from_config(config, config_path, path_overrides)

    processing = get_config_section(config, "processing")
    reconstruction = get_config_section(config, "reconstruction")
    training = get_config_section(config, "training")
    export = get_config_section(config, "export")
    kafka = get_config_section(config, "kafka")
    viewer = get_config_section(config, "viewer")

    num_frames_target = _setting(args.num_frames_target, processing, "num_frames_target", 50)
    num_downscales = _setting(args.num_downscales, processing, "num_downscales", 0)
    skip_image_extraction = _setting(
        args.skip_image_extraction, processing, "skip_image_extraction", False
    )
    num_matched = _setting(args.num_matched, reconstruction, "num_matched", 5)
    train_downscale_factor = _setting(
        args.train_downscale_factor, training, "downscale_factor", 1
    )
    max_num_iterations = _setting(
        args.max_num_iterations, training, "max_num_iterations", 30_000
    )
    compression_level = _setting(
        args.splat_compression_level, export, "splat_compression_level", 2
    )
    min_opacity = _setting(args.splat_min_opacity, export, "splat_min_opacity", 100)
    send_kafka_message = _setting(args.send_kafka_message, kafka, "send_message", True)
    start_viewer = _setting(args.start_viewer, viewer, "start_server", False)
    viewer_port = _setting(args.viewer_port, viewer, "port", 8000)

    print(f"Configuration: {config_path}", flush=True)
    print("Resolved paths:\n" + paths.display(), flush=True)
    if args.dry_run:
        print("\nDry run complete; no directories or files were created.", flush=True)
        return

    if not skip_image_extraction:
        if paths.video_path is None:
            raise PipelineConfigError(
                "Set paths.video_path, pass --video_path, or enable skip_image_extraction."
            )
        if not paths.video_path.is_file():
            raise FileNotFoundError(f"Drone video not found: {paths.video_path}")
    elif not paths.image_dir.is_dir():
        raise FileNotFoundError(f"Existing image directory not found: {paths.image_dir}")

    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    pipeline_start = perf_counter()

    if not skip_image_extraction:
        assert paths.video_path is not None
        run_stage(
            "Extracting video frames",
            [
                sys.executable,
                str(PIPELINE_ROOT / "preprocess" / "process_drone_data.py"),
                "--video_path",
                str(paths.video_path),
                "--image_dir",
                str(paths.image_dir),
                "--num_frames_target",
                str(num_frames_target),
                "--num_downscales",
                str(num_downscales),
            ],
        )
    else:
        print("\nSkipping frame extraction and using the existing images.", flush=True)

    run_stage(
        "Running hloc reconstruction",
        [
            sys.executable,
            str(PIPELINE_ROOT / "preprocess" / "run_hloc.py"),
            "--image_dir",
            str(paths.image_dir),
            "--colmap_dir",
            str(paths.colmap_dir),
            "--output_dir",
            str(paths.nerfstudio_dir),
            "--matching_method",
            str(reconstruction.get("matching_method", "sequential")),
            "--num_matched",
            str(num_matched),
            "--feature_type",
            str(reconstruction.get("feature_type", "superpoint_aachen")),
            "--matcher_type",
            str(reconstruction.get("matcher_type", "superglue")),
            "--camera_model",
            str(reconstruction.get("camera_model", "OPENCV")),
            "--single_camera",
            str(reconstruction.get("single_camera", True)).lower(),
            "--clean",
            str(reconstruction.get("clean", True)).lower(),
        ],
    )

    transforms_path = paths.nerfstudio_dir / "transforms.json"
    sparse_point_cloud_path = paths.nerfstudio_dir / "sparse_pc.ply"
    if not transforms_path.is_file() or not sparse_point_cloud_path.is_file():
        raise RuntimeError("hloc finished without producing transforms.json and sparse_pc.ply")
    check_reconstruction(
        transforms_path, paths.image_dir,
        _setting(args.min_registered_images, reconstruction, "min_registered_images", 10),
        _setting(args.min_registered_fraction, reconstruction, "min_registered_fraction", 0.5),
    )

    run_stage(
        "Training Splatfacto",
        [
            sys.executable,
            str(PIPELINE_ROOT / "train" / "train_splatfacto.py"),
            "--dataset_dir",
            str(paths.nerfstudio_dir),
            "--output_dir",
            str(paths.splatfacto_output_dir),
            "--training_config_path",
            str(paths.training_config_path),
            "--downscale_factor",
            str(train_downscale_factor),
            "--max_num_iterations",
            str(max_num_iterations),
        ],
    )

    if not paths.training_config_path.is_file():
        raise RuntimeError(f"Training finished without producing its config: {paths.training_config_path}")

    export_command = [
        sys.executable,
        str(PIPELINE_ROOT / "export" / "export_splat.py"),
        "--config",
        str(config_path),
        "--dataset_dir",
        str(paths.dataset_dir),
        "--splatfacto_output_dir",
        str(paths.splatfacto_output_dir),
        "--training_config_path",
        str(paths.training_config_path),
        "--exported_ply_path",
        str(paths.exported_ply_path),
        "--splat_output_path",
        str(paths.splat_output_path),
        "--converter_executable",
        paths.converter_executable,
        "--compression_level",
        str(compression_level),
        "--min_opacity",
        str(min_opacity),
        "--no_start_viewer",
    ]
    for option_name in ("obb_center", "obb_rotation", "obb_scale"):
        values = getattr(args, option_name)
        if values is not None:
            export_command.extend([f"--{option_name}", *(str(value) for value in values)])

    run_stage("Exporting PLY and converting it to SPLAT", export_command)
    if not paths.exported_ply_path.is_file():
        raise RuntimeError(f"PLY export was not created: {paths.exported_ply_path}")
    if not paths.splat_output_path.is_file():
        raise RuntimeError(f"SPLAT conversion was not created: {paths.splat_output_path}")

    if send_kafka_message:
        run_stage(
            "Sending the ready message to Kafka",
            [
                sys.executable,
                str(PIPELINE_ROOT / "kafka" / "producer.py"),
                "--message",
                str(paths.kafka_message_path),
            ],
        )
    else:
        print("\nSkipping the Kafka message because kafka.send_message is false.", flush=True)

    total_elapsed = perf_counter() - pipeline_start
    print(f"\nPipeline completed successfully in {total_elapsed:.2f} seconds.", flush=True)
    print("Outputs:\n" + paths.display(), flush=True)

    if start_viewer:
        print("\nStarting the web viewer. Press Ctrl+C to stop it.", flush=True)
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(PIPELINE_ROOT / "viewer" / "server.py"),
                    "--config",
                    str(config_path),
                    "--splat_path",
                    str(paths.splat_output_path),
                    "--port",
                    str(viewer_port),
                ],
                cwd=PIPELINE_ROOT,
                check=True,
            )
        except KeyboardInterrupt:
            print("\nWeb viewer stopped.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except PipelineConfigError as error:
        raise SystemExit(f"Configuration error: {error}") from error
