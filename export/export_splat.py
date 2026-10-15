"""Export a trained Splatfacto model, convert it to SPLAT, and optionally serve it."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PIPELINE_ROOT.parent
VIEWER_SPLAT_LINK = PIPELINE_ROOT / "viewer" / "splat_files" / "current.splat"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PipelineConfigError,
    PipelinePaths,
    get_config_section,
    load_pipeline_config,
)
from export.convert_ply_to_splat import convert_ply_to_splat  # noqa: E402


def build_ns_export_command(
    training_config_path: Path,
    exported_ply_path: Path,
    obb_center: Optional[Sequence[float]] = None,
    obb_rotation: Optional[Sequence[float]] = None,
    obb_scale: Optional[Sequence[float]] = None,
) -> list[str]:
    """Build the Nerfstudio command that exports the Gaussian model to PLY."""

    crop_values = {
        "obb_center": obb_center,
        "obb_rotation": obb_rotation,
        "obb_scale": obb_scale,
    }
    provided_values = [value is not None for value in crop_values.values()]
    if any(provided_values) and not all(provided_values):
        raise PipelineConfigError(
            "Export cropping requires obb_center, obb_rotation, and obb_scale together."
        )

    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "nerfstudio" / "scripts" / "exporter.py"),
        "gaussian-splat",
        "--load-config",
        str(training_config_path),
        "--output-dir",
        str(exported_ply_path.parent),
        "--output-filename",
        exported_ply_path.name,
    ]
    if all(provided_values):
        for name, values in crop_values.items():
            assert values is not None
            if len(values) != 3:
                raise PipelineConfigError(f"export.{name} must contain exactly three numbers.")
            command.extend([f"--{name.replace('_', '-')}", *(str(float(value)) for value in values)])
    return command


def export_and_convert(
    training_config_path: Path,
    exported_ply_path: Path,
    splat_output_path: Path,
    converter_executable: str,
    compression_level: int,
    min_opacity: int,
    obb_center: Optional[Sequence[float]] = None,
    obb_rotation: Optional[Sequence[float]] = None,
    obb_scale: Optional[Sequence[float]] = None,
) -> Path:
    """Export one trained checkpoint to PLY and convert the PLY to SPLAT."""

    training_config_path = training_config_path.expanduser().resolve()
    exported_ply_path = exported_ply_path.expanduser().resolve()
    splat_output_path = splat_output_path.expanduser().resolve()

    if not training_config_path.is_file():
        raise FileNotFoundError(f"Splatfacto training config not found: {training_config_path}")

    exported_ply_path.parent.mkdir(parents=True, exist_ok=True)
    export_command = build_ns_export_command(
        training_config_path,
        exported_ply_path,
        obb_center=obb_center,
        obb_rotation=obb_rotation,
        obb_scale=obb_scale,
    )
    print("\nExporting trained Splatfacto model to PLY", flush=True)
    print(f"$ {shlex.join(export_command)}", flush=True)
    subprocess.run(export_command, cwd=REPOSITORY_ROOT, check=True)

    if not exported_ply_path.is_file():
        raise RuntimeError(f"Nerfstudio export did not create: {exported_ply_path}")

    print("\nConverting exported PLY to SPLAT", flush=True)
    converted_path = convert_ply_to_splat(
        ply_path=exported_ply_path,
        output_path=splat_output_path,
        compression_level=compression_level,
        min_opacity=min_opacity,
        converter_path=converter_executable,
    )
    print(f"SPLAT file ready: {converted_path}", flush=True)
    return converted_path


def publish_to_viewer(splat_path: Path, viewer_link: Path = VIEWER_SPLAT_LINK) -> Path:
    """Point the viewer's stable static path at the generated dataset SPLAT."""

    splat_path = splat_path.expanduser().resolve()
    viewer_link = viewer_link.expanduser().absolute()
    if not splat_path.is_file():
        raise FileNotFoundError(f"Cannot publish missing SPLAT file: {splat_path}")

    viewer_link.parent.mkdir(parents=True, exist_ok=True)
    if viewer_link.is_symlink():
        viewer_link.unlink()
    elif viewer_link.exists():
        raise FileExistsError(
            f"Viewer publication path is an existing regular file: {viewer_link}"
        )
    viewer_link.symlink_to(splat_path)
    print(f"Viewer scene: {viewer_link} -> {splat_path}", flush=True)
    return viewer_link


def run_viewer(config_path: Path, splat_path: Path, port: int) -> None:
    """Run the HTTPS viewer in the foreground until the user presses Ctrl+C."""

    viewer_command = [
        sys.executable,
        str(PIPELINE_ROOT / "viewer" / "server.py"),
        "--config",
        str(config_path),
        "--splat_path",
        str(splat_path),
        "--port",
        str(port),
    ]
    print("\nStarting web viewer. Press Ctrl+C to stop it.", flush=True)
    print(f"$ {shlex.join(viewer_command)}", flush=True)
    try:
        subprocess.run(viewer_command, cwd=PIPELINE_ROOT, check=True)
    except KeyboardInterrupt:
        print("\nWeb viewer stopped.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Splatfacto to PLY, convert it to SPLAT, and optionally start the web viewer."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset_dir", type=Path, help="Override paths.dataset_dir.")
    parser.add_argument("--splatfacto_output_dir", type=Path)
    parser.add_argument("--training_config_path", type=Path)
    parser.add_argument("--export_dir", type=Path)
    parser.add_argument("--exported_ply_path", type=Path)
    parser.add_argument("--splat_output_path", type=Path)
    parser.add_argument("--converter_executable")
    parser.add_argument("--compression_level", type=int)
    parser.add_argument("--min_opacity", type=int)
    parser.add_argument("--obb_center", type=float, nargs=3)
    parser.add_argument("--obb_rotation", type=float, nargs=3)
    parser.add_argument("--obb_scale", type=float, nargs=3)
    parser.add_argument("--port", type=int)

    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument("--start_viewer", dest="start_viewer", action="store_true")
    viewer_group.add_argument("--no_start_viewer", dest="start_viewer", action="store_false")
    parser.set_defaults(start_viewer=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_pipeline_config(config_path)
    paths = PipelinePaths.from_config(
        config,
        config_path,
        {
            "dataset_dir": args.dataset_dir,
            "splatfacto_output_dir": args.splatfacto_output_dir,
            "training_config_path": args.training_config_path,
            "export_dir": args.export_dir,
            "exported_ply_path": args.exported_ply_path,
            "splat_output_path": args.splat_output_path,
            "converter_executable": args.converter_executable,
        },
    )

    export_config = get_config_section(config, "export")
    viewer_config = get_config_section(config, "viewer")
    compression_level = (
        args.compression_level
        if args.compression_level is not None
        else export_config.get("splat_compression_level", 2)
    )
    min_opacity = (
        args.min_opacity
        if args.min_opacity is not None
        else export_config.get("splat_min_opacity", 100)
    )
    obb_center = args.obb_center if args.obb_center is not None else export_config.get("obb_center")
    obb_rotation = (
        args.obb_rotation if args.obb_rotation is not None else export_config.get("obb_rotation")
    )
    obb_scale = args.obb_scale if args.obb_scale is not None else export_config.get("obb_scale")
    start_viewer = (
        args.start_viewer
        if args.start_viewer is not None
        else viewer_config.get("start_server", False)
    )
    port = args.port if args.port is not None else viewer_config.get("port", 8000)

    export_command = build_ns_export_command(
        paths.training_config_path,
        paths.exported_ply_path,
        obb_center=obb_center,
        obb_rotation=obb_rotation,
        obb_scale=obb_scale,
    )
    print("Resolved export paths:", flush=True)
    print(f"  training_config_path: {paths.training_config_path}", flush=True)
    print(f"  exported_ply_path: {paths.exported_ply_path}", flush=True)
    print(f"  splat_output_path: {paths.splat_output_path}", flush=True)
    print(f"  viewer_splat_link: {VIEWER_SPLAT_LINK}", flush=True)
    print(f"  ns_export_command: {shlex.join(export_command)}", flush=True)
    if args.dry_run:
        print("Dry run complete; no files were created.", flush=True)
        return

    converted_path = export_and_convert(
        training_config_path=paths.training_config_path,
        exported_ply_path=paths.exported_ply_path,
        splat_output_path=paths.splat_output_path,
        converter_executable=paths.converter_executable,
        compression_level=compression_level,
        min_opacity=min_opacity,
        obb_center=obb_center,
        obb_rotation=obb_rotation,
        obb_scale=obb_scale,
    )
    publish_to_viewer(converted_path)
    if start_viewer:
        run_viewer(config_path=config_path, splat_path=converted_path, port=port)


if __name__ == "__main__":
    try:
        main()
    except PipelineConfigError as error:
        raise SystemExit(f"Configuration error: {error}") from error
