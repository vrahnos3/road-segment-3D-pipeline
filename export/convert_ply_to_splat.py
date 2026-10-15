import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_GSCONVERTER = "3dgsconverter"


def resolve_converter(converter_path: str | Path) -> str:
    """Resolve either an executable name on PATH or an explicit file path."""
    converter = str(converter_path)
    if "/" not in converter:
        resolved = shutil.which(converter)
        if resolved is not None:
            return resolved
    else:
        explicit_path = Path(converter).expanduser().resolve()
        if explicit_path.is_file():
            return str(explicit_path)

    raise FileNotFoundError(
        f"3dgsconverter was not found: {converter}\n"
        "Set paths.converter_executable in config.yaml or pass --converter_path."
    )


def convert_ply_to_splat(
    ply_path: Path,
    output_path: Path,
    compression_level: int = 1,
    min_opacity: int = 5,
    converter_path: str | Path = DEFAULT_GSCONVERTER,
) -> Path:
    ply_path = ply_path.resolve()
    output_path = output_path.resolve()

    converter = resolve_converter(converter_path)

    converter_cmd = [
        converter,
        "-i", str(ply_path),
        "-o", str(output_path),
        "-f", "splat",
        "--compression_level", str(compression_level),
        "--min_opacity", str(min_opacity),
        "--force",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(converter_cmd, check=True)

    if not output_path.exists():
        raise RuntimeError(f"Output file was not created: {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Gaussian Splatting PLY file to SPLAT format.")
    parser.add_argument("--ply_path", required=True, type=Path, help="Input Gaussian Splatting PLY file")
    parser.add_argument("--output_path", required=True, type=Path, help="Output SPLAT file")
    parser.add_argument(
        "--converter_path",
        default=DEFAULT_GSCONVERTER,
        help="3dgsconverter executable path or command name (default: 3dgsconverter)",
    )
    parser.add_argument(
        "--compression_level", type=int, default=2, help="3dgsconverter compression level"
    )
    parser.add_argument("--min_opacity", type=int, default=100, help="Minimum exported opacity")
    args = parser.parse_args()

    converted_path = convert_ply_to_splat(
        ply_path=args.ply_path,
        output_path=args.output_path,
        compression_level=args.compression_level,
        min_opacity=args.min_opacity,
        converter_path=args.converter_path,
    )

    print(f"File {args.ply_path} converted to {converted_path}")


if __name__ == "__main__":
    main()
