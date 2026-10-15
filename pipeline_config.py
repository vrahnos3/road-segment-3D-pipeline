"""Configuration loading and path resolution for the iDriving pipeline."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "config" / "config.yaml"


class PipelineConfigError(ValueError):
    """Raised when the pipeline YAML has an invalid value or structure."""


def load_pipeline_config(config_path: Path) -> dict[str, Any]:
    """Load a pipeline YAML file and return its top-level mapping."""

    config_path = config_path.expanduser().resolve()
    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PipelineConfigError(f"Configuration file not found: {config_path}") from error
    except yaml.YAMLError as error:
        raise PipelineConfigError(f"Invalid YAML in {config_path}: {error}") from error

    if raw_config is None:
        return {}
    if not isinstance(raw_config, dict):
        raise PipelineConfigError(f"The top level of {config_path} must be a mapping.")
    return raw_config


def get_config_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return one optional mapping from the configuration."""

    section = config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise PipelineConfigError(f"Configuration section '{name}' must be a mapping.")
    return section


def publish_training_config(
    source: Path, destination: Path, checkpoint_dir: Path, final_step: int
) -> None:
    """Copy the completed run's config to the stable path used by the exporter.

    The YAML retains Nerfstudio's timestamp and original checkpoint location.
    Never select a different run merely because its config is newer.
    """
    if not source.is_file():
        raise FileNotFoundError(f"Training did not produce its config: {source}")
    final_checkpoint = checkpoint_dir / f"step-{final_step:09d}.ckpt"
    if not final_checkpoint.is_file() or not final_checkpoint.stat().st_size:
        raise RuntimeError(
            f"Training did not complete: final checkpoint missing or empty: {final_checkpoint}. "
            "Export and the Kafka ready notification must not run."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    print(f"Training config for export: {destination}", flush=True)


def _resolve_path(value: str | Path, relative_to: Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if not expanded.is_absolute():
        expanded = relative_to / expanded
    return expanded.resolve()


@dataclass(frozen=True)
class PipelinePaths:
    """All input and output paths used by the complete pipeline."""

    video_path: Optional[Path]
    dataset_dir: Path
    image_dir: Path
    colmap_dir: Path
    nerfstudio_dir: Path
    splatfacto_output_dir: Path
    training_config_path: Path
    export_dir: Path
    exported_ply_path: Path
    splat_output_path: Path
    kafka_message_path: Path
    converter_executable: str

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        config_path: Path,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "PipelinePaths":
        """Resolve paths, applying CLI overrides before dataset-relative defaults.

        Relative YAML paths are resolved from the YAML file's directory. Relative
        command-line overrides are resolved from the current working directory.
        """

        path_config = get_config_section(config, "paths")
        overrides = overrides or {}
        yaml_base = config_path.expanduser().resolve().parent
        cli_base = Path.cwd()

        def explicit(name: str, yaml_relative_to: Path = yaml_base) -> Optional[Path]:
            override = overrides.get(name)
            if override is not None:
                return _resolve_path(override, cli_base)
            value = path_config.get(name)
            if value in (None, ""):
                return None
            return _resolve_path(value, yaml_relative_to)

        dataset_dir = explicit("dataset_dir")
        if dataset_dir is None:
            raise PipelineConfigError(
                "Set paths.dataset_dir in the YAML file or pass --dataset_dir."
            )

        # Relative output paths in YAML belong to the dataset. This lets users
        # rename an output folder without repeating the complete dataset path.
        image_dir = explicit("image_dir", dataset_dir) or dataset_dir / "images"
        colmap_dir = explicit("colmap_dir", dataset_dir) or dataset_dir / "colmap"
        nerfstudio_dir = explicit("nerfstudio_dir", dataset_dir) or dataset_dir
        splatfacto_output_dir = (
            explicit("splatfacto_output_dir", dataset_dir) or dataset_dir / "splatfacto_output"
        )
        training_config_path = (
            explicit("training_config_path", dataset_dir) or splatfacto_output_dir / "config.yml"
        )
        export_dir = explicit("export_dir", dataset_dir) or dataset_dir / "export"
        exported_ply_path = explicit("exported_ply_path", dataset_dir) or export_dir / "splat.ply"
        splat_output_path = explicit("splat_output_path", dataset_dir) or export_dir / "splat.splat"
        kafka_message_path = explicit("kafka_message_path") or (
            PIPELINE_ROOT / "kafka" / "templates" / "3d_reconstruction_ready_v2.json"
        )

        converter_override = overrides.get("converter_executable")
        converter_value = (
            converter_override
            if converter_override is not None
            else path_config.get("converter_executable", "3dgsconverter")
        )
        if not isinstance(converter_value, (str, Path)) or not str(converter_value).strip():
            raise PipelineConfigError("paths.converter_executable must be a non-empty string.")
        converter_executable = str(converter_value)
        if "/" in converter_executable or converter_executable.startswith("."):
            converter_base = cli_base if converter_override is not None else yaml_base
            converter_executable = str(_resolve_path(converter_executable, converter_base))

        return cls(
            video_path=explicit("video_path"),
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            colmap_dir=colmap_dir,
            nerfstudio_dir=nerfstudio_dir,
            splatfacto_output_dir=splatfacto_output_dir,
            training_config_path=training_config_path,
            export_dir=export_dir,
            exported_ply_path=exported_ply_path,
            splat_output_path=splat_output_path,
            kafka_message_path=kafka_message_path,
            converter_executable=converter_executable,
        )

    def display(self) -> str:
        """Return a readable summary for logs and dry runs."""

        values = (
            ("video_path", self.video_path),
            ("dataset_dir", self.dataset_dir),
            ("image_dir", self.image_dir),
            ("colmap_dir", self.colmap_dir),
            ("nerfstudio_dir", self.nerfstudio_dir),
            ("splatfacto_output_dir", self.splatfacto_output_dir),
            ("training_config_path", self.training_config_path),
            ("export_dir", self.export_dir),
            ("exported_ply_path", self.exported_ply_path),
            ("splat_output_path", self.splat_output_path),
            ("kafka_message_path", self.kafka_message_path),
            ("converter_executable", self.converter_executable),
        )
        return "\n".join(f"  {name}: {value}" for name, value in values)
