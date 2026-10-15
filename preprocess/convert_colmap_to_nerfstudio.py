"""Prepare an existing COLMAP/VGGT reconstruction for Splatfacto training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Optional


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PIPELINE_ROOT.parent
for import_root in (REPOSITORY_ROOT, PIPELINE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from pipeline_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PipelineConfigError,
    PipelinePaths,
    load_pipeline_config,
)


def find_model_dir(recon_dir: Path) -> tuple[Path, str]:
    """Find a complete binary or text COLMAP model in common folder layouts."""

    recon_dir = recon_dir.expanduser().resolve()
    candidates = [recon_dir, recon_dir / "sparse", recon_dir / "sparse" / "0", recon_dir / "0"]
    for candidate in candidates:
        for extension in (".bin", ".txt"):
            if all((candidate / f"{name}{extension}").is_file() for name in ("cameras", "images", "points3D")):
                return candidate, extension
    raise FileNotFoundError(
        f"No complete COLMAP model found under {recon_dir}. Expected cameras.bin, "
        "images.bin, and points3D.bin (or all three .txt files). Pass --recon_dir "
        "pointing directly to the folder containing these files."
    )


def convert_reconstruction(
    recon_dir: Path,
    image_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> int:
    """Export camera poses/intrinsics and points without rerunning reconstruction.

    Images are referenced at their original locations, not copied or renamed.
    Both generated files are staged before replacing the final output files.
    """

    model_dir, extension = find_model_dir(recon_dir)
    image_dir = image_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    for filename in ("transforms.json", "sparse_pc.ply"):
        if (output_dir / filename).exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir / filename}. Use --overwrite to regenerate it."
            )

    # Import the training environment's dependencies only when actually converting.
    from PIL import Image as PILImage

    from nerfstudio.data.utils.colmap_parsing_utils import read_model, write_model
    from nerfstudio.process_data.colmap_utils import colmap_to_json, parse_colmap_camera_params

    cameras, images, points = read_model(model_dir, ext=extension)
    if not cameras or not images:
        raise ValueError("The COLMAP model must contain cameras and registered images.")
    if not points:
        raise ValueError("The COLMAP model has no 3D points to initialize Splatfacto.")

    for camera in cameras.values():
        parse_colmap_camera_params(camera)

    frame_paths = {}
    for image_id, image in images.items():
        if image.camera_id not in cameras:
            raise ValueError(f"Image {image.name} references missing camera {image.camera_id}.")
        image_path = image_dir / image.name
        if not image_path.is_file():
            raise FileNotFoundError(f"Registered image not found: {image_path}")
        camera = cameras[image.camera_id]
        with PILImage.open(image_path) as source_image:
            if source_image.size != (camera.width, camera.height):
                raise ValueError(
                    f"Image size mismatch for {image_path}: image is {source_image.size}, "
                    f"COLMAP camera expects {(camera.width, camera.height)}. Use the same "
                    "resized/cropped images used to build the VGGT reconstruction."
                )
        frame_paths[image_id] = Path(os.path.relpath(image_path, output_dir)).as_posix()

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".colmap_export_", dir=output_dir) as temporary_dir:
        staging_dir = Path(temporary_dir)
        binary_model_dir = model_dir
        if extension == ".txt":
            binary_model_dir = staging_dir / "binary_model"
            binary_model_dir.mkdir()
            write_model(cameras, images, points, binary_model_dir, ext=".bin")

        # Export per-frame intrinsics: VGGT can use a different camera per image.
        count = colmap_to_json(
            recon_dir=binary_model_dir,
            output_dir=staging_dir,
            use_single_camera_mode=False,
        )
        transforms_path = staging_dir / "transforms.json"
        metadata = json.loads(transforms_path.read_text(encoding="utf-8"))
        for frame in metadata["frames"]:
            frame["file_path"] = frame_paths[frame["colmap_im_id"]]
        transforms_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

        for filename in ("sparse_pc.ply", "transforms.json"):
            os.replace(staging_dir / filename, output_dir / filename)

    print(f"Converted {count} registered images and {len(points)} 3D points.")
    print(f"Camera metadata: {output_dir / 'transforms.json'}")
    print(f"Sparse point cloud: {output_dir / 'sparse_pc.ply'}")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert existing COLMAP/VGGT output to Nerfstudio files without running hloc or COLMAP."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset_dir", type=Path, help="Override the configured dataset root.")
    parser.add_argument("--recon_dir", type=Path, help="COLMAP model folder, or its parent colmap/sparse folder.")
    parser.add_argument("--image_dir", type=Path, help="The images used to create the reconstruction.")
    parser.add_argument("--output_dir", type=Path, help="Folder for transforms.json and sparse_pc.ply.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing metadata and sparse point cloud.")
    parser.add_argument("--dry_run", action="store_true", help="Check the model location without exporting anything.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_pipeline_config(config_path)
    paths = PipelinePaths.from_config(
        config,
        config_path,
        {"dataset_dir": args.dataset_dir, "image_dir": args.image_dir, "nerfstudio_dir": args.output_dir},
    )
    recon_dir = args.recon_dir if args.recon_dir is not None else paths.colmap_dir
    model_dir, extension = find_model_dir(recon_dir)
    print(f"Reconstruction: {model_dir} ({extension})")
    print(f"Images: {paths.image_dir}")
    print(f"Output dataset: {paths.nerfstudio_dir}")
    if args.dry_run:
        print("Dry run complete; no files were created.")
        return
    convert_reconstruction(model_dir, paths.image_dir, paths.nerfstudio_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except (PipelineConfigError, FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(f"Conversion error: {error}") from error
