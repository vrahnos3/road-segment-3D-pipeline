from rich.console import Console
import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Literal, Optional
import pycolmap
from datetime import timedelta
from time import perf_counter
from hloc import extract_features, match_features, pairs_from_exhaustive, pairs_from_retrieval, reconstruction

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary
from nerfstudio.process_data.colmap_utils import colmap_to_json


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


def get_image_list(image_dir: Path) -> List[str]:
    """Return image paths relative to image_dir, sorted for repeatable pairing."""
    return sorted(
        path.relative_to(image_dir).as_posix()
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_sequential_pairs(image_list: List[str], pairs_path: Path, num_matched: int) -> None:
    """Write pairs between nearby images in sorted order."""
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("w", encoding="utf-8") as f:
        for i, image_name in enumerate(image_list):
            for j in range(i + 1, min(i + num_matched + 1, len(image_list))):
                f.write(f"{image_name} {image_list[j]}\n")


def export_nerfstudio_dataset(
    image_dir: Path,
    colmap_dir: Path,
    output_dir: Path,
    use_single_camera_mode: bool,
) -> None:
    """Export hloc/COLMAP reconstruction to Nerfstudio transforms.json and sparse point cloud."""
    recon_dir = colmap_dir / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)

    CONSOLE.log("[bold yellow]Exporting Nerfstudio transforms.json and sparse_pc.ply...")
    num_registered = colmap_to_json(
        recon_dir=recon_dir,
        output_dir=output_dir,
        use_single_camera_mode=use_single_camera_mode,
    )
    CONSOLE.log(
        f"[bold green]Done exporting {output_dir / 'transforms.json'} "
        f"and {output_dir / 'sparse_pc.ply'} for {num_registered} registered images."
    )


def run_hloc(
    image_dir: Path,
    colmap_dir: Path,
    output_dir: Optional[Path] = None,
    camera_model: Literal["OPENCV", "PINHOLE", "SIMPLE_PINHOLE"] = "OPENCV",
    verbose: bool = True,
    matching_method: Literal["retrieval", "vocab_tree", "exhaustive", "sequential"] = "sequential",
    feature_type: Literal[
        "sift", "superpoint_aachen", "superpoint_max", "superpoint_inloc", "r2d2", "d2net-ss", "sosnet", "disk"
    ] = "superpoint_aachen",
    matcher_type: Literal[
        "superglue",
        "superglue-fast",
        "NN-superpoint",
        "NN-ratio",
        "NN-mutual",
        "adalam",
        "disk+lightglue",
        "superpoint+lightglue",
    ] = "superglue",
    num_matched: int = 50,
    use_single_camera_mode: bool = True,
    clean: bool = True,
    export_nerfstudio: bool = True,
) -> None:
    """Runs hloc on the images and writes a COLMAP-compatible sparse reconstruction.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        output_dir: Path where transforms.json and sparse_pc.ply will be written.
        camera_model: COLMAP camera model to use.
        verbose: If True, logs the output of hloc/COLMAP calls.
        matching_method: Matching method to use.
        feature_type: Local feature type used by hloc.
        matcher_type: Feature matcher used by hloc.
        num_matched: Number of retrieval matches per image for non-exhaustive matching.
        use_single_camera_mode: If True, use one shared camera for all images.
        clean: If True, remove previous hloc outputs in colmap_dir before running.
        export_nerfstudio: If True, export transforms.json and sparse_pc.ply.
    """

    image_dir = image_dir.resolve()
    colmap_dir = colmap_dir.resolve()
    output_dir = image_dir.parent.resolve() if output_dir is None else output_dir.resolve()
    colmap_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = colmap_dir / "sparse"
    sfm_dir = sparse_dir / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = colmap_dir / "pairs-sfm.txt"
    features_path = colmap_dir / "features.h5"
    matches_path = colmap_dir / "matches.h5"

    if clean:
        for path in [
            features_path,
            matches_path,
            pairs_path,
            colmap_dir / "global-feats-netvlad.h5",
            output_dir / "transforms.json",
            output_dir / "sparse_pc.ply",
        ]:
            # deletes a file or symbolic link
            path.unlink(missing_ok=True)
        if sfm_dir.exists():
            # recursively deletes that directory and everything inside it
            shutil.rmtree(sfm_dir)

    

    image_list = get_image_list(image_dir)
    feature_conf = extract_features.confs[feature_type]
    matcher_conf = match_features.confs[matcher_type]

    CONSOLE.log("[bold yellow]Extracting hloc local features...")
    extract_features.main(feature_conf, image_dir, image_list=image_list, feature_path=features_path)

    CONSOLE.log("[bold yellow]Generating hloc image pairs...")
    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(pairs_path, image_list=image_list)
    elif matching_method == "sequential":
        write_sequential_pairs(image_list, pairs_path, num_matched=num_matched)
    elif matching_method in {"retrieval", "vocab_tree"}:
        retrieval_conf = extract_features.confs["netvlad"]
        retrieval_path = extract_features.main(retrieval_conf, image_dir, colmap_dir, image_list=image_list)
        pairs_from_retrieval.main(retrieval_path, pairs_path, num_matched=min(len(image_list), num_matched))

    CONSOLE.log("[bold yellow]Matching hloc image pairs...")
    match_features.main(matcher_conf, pairs_path, features=features_path, matches=matches_path)

    image_options = {"camera_model": camera_model}
    camera_mode = pycolmap.CameraMode.SINGLE if use_single_camera_mode else pycolmap.CameraMode.PER_IMAGE

    CONSOLE.log("[bold yellow]Building sparse reconstruction with hloc...")
    
    reconstruction.main(
        sfm_dir,
        image_dir,
        pairs_path,
        features_path,
        matches_path,
        image_list=image_list,
        camera_mode=camera_mode,
        image_options=image_options,
        verbose=verbose,
    )

    CONSOLE.log("[bold green]:tada: Done running hloc.")

    print_hloc_registered_percentage(image_dir, sfm_dir)
    if export_nerfstudio:
        export_nerfstudio_dataset(
            image_dir=image_dir,
            colmap_dir=colmap_dir,
            output_dir=output_dir,
            use_single_camera_mode=use_single_camera_mode,
        )


def count_input_images(image_dir: Path) -> int:
    """Count image files hloc was asked to process."""
    return sum(1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def get_registered_image_count(recon_dir: Path) -> int:
    """Read COLMAP images.bin and return the number of registered images."""
    images_bin = recon_dir / "images.bin"
    if not images_bin.exists():
        return 0
    return len(read_images_binary(images_bin))


def print_hloc_registered_percentage(image_dir: Path, recon_dir: Path) -> None:
    """Print the registered-image percentage for the hloc reconstruction."""
    num_input_images = count_input_images(image_dir)
    num_registered_images = get_registered_image_count(recon_dir)

    if num_input_images == 0:
        CONSOLE.print("[bold red]No input images found, cannot compute hloc registration percentage.")
        return

    percentage = num_registered_images / num_input_images * 100
    CONSOLE.print(
        f"[bold green]hloc registered {num_registered_images}/{num_input_images} images "
        f"({percentage:.2f}%)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hloc SfM on a folder of drone images.")
    parser.add_argument(
        "-i", "--image_dir", required=True, type=Path, help="Directory containing extracted images"
    )
    parser.add_argument(
        "-c",
        "--colmap_dir",
        required=True,
        type=Path,
        help="Directory where hloc/COLMAP-compatible outputs will be written",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=None,
        help="Directory where transforms.json and sparse_pc.ply will be written. Defaults to image-dir parent.",
    )
    parser.add_argument(
        "--camera_model",
        default="OPENCV",
        choices=["OPENCV", "PINHOLE", "SIMPLE_PINHOLE"],
        help="COLMAP camera model (default: OPENCV)",
    )
    parser.add_argument(
        "--matching_method",
        default="sequential",
        choices=["retrieval", "vocab_tree", "exhaustive", "sequential"],
        help="hloc image-pair generation method. retrieval uses NetVLAD; vocab_tree is kept as an alias (default: sequential)",
    )
    parser.add_argument(
        "--feature_type",
        default="superpoint_aachen",
        choices=["sift", "superpoint_aachen", "superpoint_max", "superpoint_inloc", "r2d2", "d2net-ss", "sosnet", "disk"],
        help="hloc local feature type (default: superpoint_aachen)",
    )
    parser.add_argument(
        "--matcher_type",
        default="superglue",
        choices=[
            "superglue",
            "superglue-fast",
            "NN-superpoint",
            "NN-ratio",
            "NN-mutual",
            "adalam",
            "disk+lightglue",
            "superpoint+lightglue",
        ],
        help="hloc matcher type (default: superglue)",
    )
    parser.add_argument(
        "--num_matched",
        type=int,
        default=50,
        help="Number of pairs per image for retrieval/sequential matching (default: 50)",
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="Show hloc command output")
    parser.add_argument(
        "--single_camera",
        type=str_to_bool,
        default=True,
        help="Use a single shared camera for all images: true or false (default: true)",
    )
    parser.add_argument(
        "--clean",
        type=str_to_bool,
        default=True,
        help="Delete previous hloc outputs in the colmap directory before running: true or false (default: true)",
    )
    parser.add_argument(
        "--export_nerfstudio",
        type=str_to_bool,
        default=True,
        help="Export transforms.json and sparse_pc.ply after reconstruction: true or false (default: true)",
    )
    args = parser.parse_args()

    run_hloc(
        image_dir=args.image_dir,
        colmap_dir=args.colmap_dir,
        output_dir=args.output_dir,
        camera_model=args.camera_model,
        verbose=args.verbose,
        matching_method=args.matching_method,
        feature_type=args.feature_type,
        matcher_type=args.matcher_type,
        num_matched=args.num_matched,
        use_single_camera_mode=args.single_camera,
        clean=args.clean,
        export_nerfstudio=args.export_nerfstudio,
    )


if __name__ == "__main__":
    start_time = perf_counter()

    main()

    end_time = perf_counter()
    total_time = end_time - start_time
    formatted_time = str(timedelta(seconds=total_time))
    print(f"Execution time: {formatted_time}")
