from rich.console import Console
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Literal
from datetime import timedelta
from time import perf_counter
if __package__:
    from .utils import run_command
else:
    from utils import run_command

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary


CONSOLE = Console(width=120)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def str_to_bool(value: str) -> bool:
    """Parse a command-line true/false value."""
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


def run_colmap(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: Literal["OPENCV", "PINHOLE", "SIMPLE_PINHOLE"] = "OPENCV",
    gpu: bool = True,
    verbose: bool = True,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential", "spatial"] = "sequential",
    refine_intrinsics: bool = True,
    use_pose_prior_mapper: bool = False,
    colmap_cmd: str = "colmap",
) -> None:
    """Runs COLMAP on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        gpu: If True, use GPU.
        verbose: If True, logs the output of the command.
        matching_method: Matching method to use.
        refine_intrinsics: If True, refine intrinsics.
        use_pose_prior_mapper: If True, run COLMAP's pose_prior_mapper instead of mapper.
        colmap_cmd: Path to the COLMAP executable.
    """

    image_dir = image_dir.resolve()
    colmap_dir = colmap_dir.resolve()
    colmap_dir.mkdir(parents=True, exist_ok=True)
    colmap_database_path = colmap_dir / "database.db"
    colmap_database_path.unlink(missing_ok=True) # Delete the file at colmap_database_path, but do not crash if it does not exist.
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Feature extraction
    feature_extractor_cmd = [
        f"{colmap_cmd} feature_extractor",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        "--ImageReader.single_camera 1",
        f"--ImageReader.camera_model {camera_model}",
        f"--SiftExtraction.use_gpu {int(gpu)}",
    ]
    
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    CONSOLE.log("[bold yellow] Running COLMAP feature extractor...")
    run_command(feature_extractor_cmd, verbose=verbose)
    CONSOLE.log("[bold green] Done extracting COLMAP features.")

    # Feature matching
    feature_matcher_cmd = [
        f"{colmap_cmd} {matching_method}_matcher",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--SiftMatching.use_gpu {int(gpu)}",
    ]

    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    CONSOLE.log("[bold yellow] Running COLMAP feature matcher...")
    run_command(feature_matcher_cmd, verbose=verbose)
    CONSOLE.log("[bold green] Done matching COLMAP features.")

    # Bundle adjustment
    mapper_name = "pose_prior_mapper" if use_pose_prior_mapper else "mapper"
    mapper_cmd = [
        f"{colmap_cmd} {mapper_name}",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
    ]

    mapper_cmd = " ".join(mapper_cmd)

    CONSOLE.log(f"[bold yellow]Running COLMAP {mapper_name}... (This may take a while)")
    run_command(mapper_cmd, verbose=verbose)
    CONSOLE.log(f"[bold green]:tada: Done COLMAP {mapper_name}.")

    if refine_intrinsics:
        CONSOLE.log("[bold yellow]Refine intrinsics...")
        bundle_adjuster_cmd = [
            f"{colmap_cmd} bundle_adjuster",
            f"--input_path {sparse_dir}/0",
            f"--output_path {sparse_dir}/0",
            "--BundleAdjustment.refine_principal_point 1",
        ]
        run_command(" ".join(bundle_adjuster_cmd), verbose=verbose)
        CONSOLE.log("[bold green] Done refining intrinsics.")

    print_colmap_registered_percentage(image_dir, sparse_dir / "0")


def count_input_images(image_dir: Path) -> int:
    """Count image files COLMAP was asked to process."""
    return sum(1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def get_registered_image_count(recon_dir: Path) -> int:
    """Read COLMAP images.bin and return the number of registered images."""
    images_bin = recon_dir / "images.bin"
    if not images_bin.exists():
        return 0
    return len(read_images_binary(images_bin))


def print_colmap_registered_percentage(image_dir: Path, recon_dir: Path) -> None:
    """Print the same registered-image percentage Nerfstudio reports."""
    num_input_images = count_input_images(image_dir)
    num_registered_images = get_registered_image_count(recon_dir)

    if num_input_images == 0:
        CONSOLE.print("[bold red]No input images found, cannot compute COLMAP registration percentage.")
        return

    percentage = num_registered_images / num_input_images * 100
    CONSOLE.print(
        f"[bold green]COLMAP registered {num_registered_images}/{num_input_images} images "
        f"({percentage:.2f}%)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COLMAP on a folder of drone images.")
    parser.add_argument("-i", "--image_dir", required=True, type=Path, help="Directory containing extracted images")
    parser.add_argument("-c", "--colmap_dir", required=True, type=Path, help="Directory where COLMAP outputs will be written")
    parser.add_argument(
        "--camera_model",
        default="OPENCV",
        choices=["OPENCV", "PINHOLE", "SIMPLE_PINHOLE"],
        help="COLMAP camera model (default: OPENCV)",
    )
    parser.add_argument(
        "--matching_method",
        default="sequential",
        choices=["vocab_tree", "exhaustive", "sequential", "spatial"],
        help="COLMAP matcher to use. Use spatial for GPS-tagged images (default: sequential)",
    )
    parser.add_argument("--gpu", type=str_to_bool, default=True, help="Use GPU for SIFT extraction/matching: true or false (default: true)")
    parser.add_argument("--verbose", action="store_true", default=True, help="Show COLMAP command output")
    parser.add_argument("--refine_intrinsics", type=str_to_bool, default=True, help="Run COLMAP bundle_adjuster refinement: true or false (default: true)")
    parser.add_argument("--pose_prior_mapper", action="store_true", help="Use COLMAP pose_prior_mapper instead of mapper. Useful with GPS EXIF priors.")
    parser.add_argument("--colmap_cmd", default="colmap", help="COLMAP executable path/name")
    args = parser.parse_args()

    run_colmap(
        image_dir=args.image_dir,
        colmap_dir=args.colmap_dir,
        camera_model=args.camera_model,
        gpu=args.gpu,
        verbose=args.verbose,
        matching_method=args.matching_method,
        refine_intrinsics=args.refine_intrinsics,
        use_pose_prior_mapper=args.pose_prior_mapper,
        colmap_cmd=args.colmap_cmd,
    )


if __name__ == "__main__":
    start_time = perf_counter()

    main()

    end_time = perf_counter()
    total_time = end_time - start_time
    formatted_time = str(timedelta(seconds=total_time))
    print(f"Execution time: {formatted_time}")
