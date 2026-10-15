import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace


# Allow the tests to run both from the repository root and from inside
# iDriving_pipeline without installing iDriving_pipeline as a package.
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_config import PipelineConfigError, PipelinePaths, load_pipeline_config  # noqa: E402
from export.export_splat import build_ns_export_command, publish_to_viewer  # noqa: E402
from run_pipeline import build_parser  # noqa: E402
from viewer.server import SecureRequestHandler  # noqa: E402


class PipelineConfigTests(unittest.TestCase):
    def test_dataset_dir_provides_all_output_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            tmp_path = Path(temporary_dir)
            dataset_dir = tmp_path / "road_dataset"
            config_path = tmp_path / "config" / "config.yaml"
            paths = PipelinePaths.from_config(
                {"paths": {"dataset_dir": str(dataset_dir), "video_path": "../input.mp4"}},
                config_path,
            )

            self.assertEqual(paths.dataset_dir, dataset_dir)
            self.assertEqual(paths.image_dir, dataset_dir / "images")
            self.assertEqual(paths.colmap_dir, dataset_dir / "colmap")
            self.assertEqual(paths.nerfstudio_dir, dataset_dir)
            self.assertEqual(paths.splatfacto_output_dir, dataset_dir / "splatfacto_output")
            self.assertEqual(paths.training_config_path, dataset_dir / "splatfacto_output" / "config.yml")
            self.assertEqual(paths.exported_ply_path, dataset_dir / "export" / "splat.ply")
            self.assertEqual(paths.splat_output_path, dataset_dir / "export" / "splat.splat")
            self.assertEqual(paths.video_path, tmp_path / "input.mp4")

    def test_individual_paths_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            tmp_path = Path(temporary_dir)
            config_path = tmp_path / "config.yaml"
            paths = PipelinePaths.from_config(
                {
                    "paths": {
                        "dataset_dir": "configured-dataset",
                        "image_dir": "configured-images",
                    }
                },
                config_path,
                {
                    "dataset_dir": tmp_path / "cli-dataset",
                    "splat_output_path": tmp_path / "published" / "scene.splat",
                },
            )

            self.assertEqual(paths.dataset_dir, tmp_path / "cli-dataset")
            self.assertEqual(paths.image_dir, tmp_path / "cli-dataset" / "configured-images")
            self.assertEqual(paths.colmap_dir, tmp_path / "cli-dataset" / "colmap")
            self.assertEqual(paths.splat_output_path, tmp_path / "published" / "scene.splat")

    def test_relative_splatfacto_output_is_inside_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            dataset_dir = Path(temporary_dir) / "dataset100"
            config_path = Path(temporary_dir) / "config" / "config.yaml"
            paths = PipelinePaths.from_config(
                {
                    "paths": {
                        "dataset_dir": str(dataset_dir),
                        "splatfacto_output_dir": "splatfacto_output_hloc_matched5",
                    }
                },
                config_path,
            )

            expected_output = dataset_dir / "splatfacto_output_hloc_matched5"
            self.assertEqual(paths.splatfacto_output_dir, expected_output)
            self.assertEqual(paths.training_config_path, expected_output / "config.yml")

    def test_loader_rejects_non_mapping_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_path = Path(temporary_dir) / "config.yaml"
            config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

            with self.assertRaisesRegex(PipelineConfigError, "top level"):
                load_pipeline_config(config_path)

    def test_pipeline_cli_uses_underscore_argument_names(self) -> None:
        args = build_parser().parse_args(
            ["--dataset_dir", "/tmp/dataset", "--video_path", "/tmp/video.mp4", "--dry_run"]
        )

        self.assertEqual(args.dataset_dir, Path("/tmp/dataset"))
        self.assertEqual(args.video_path, Path("/tmp/video.mp4"))
        self.assertTrue(args.dry_run)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--dataset-dir", "/tmp/dataset"])

    def test_ns_export_command_uses_config_and_requested_ply_name(self) -> None:
        training_config = Path("/data/training/config.yml")
        ply_path = Path("/data/export/road_scene.ply")

        command = build_ns_export_command(training_config, ply_path)

        self.assertIn(str(training_config), command)
        self.assertIn(str(ply_path.parent), command)
        self.assertEqual(command[-1], ply_path.name)

    def test_ns_export_command_includes_oriented_bounding_box(self) -> None:
        command = build_ns_export_command(
            Path("/data/training/config.yml"),
            Path("/data/export/splat.ply"),
            obb_center=[0.0, 0.0, 0.0],
            obb_rotation=[0.0, 0.0, 0.0],
            obb_scale=[1.0, 5.0, 1.0],
        )

        scale_index = command.index("--obb-scale")
        self.assertEqual(command[scale_index + 1 : scale_index + 4], ["1.0", "5.0", "1.0"])

    def test_ns_export_crop_requires_all_three_obb_values(self) -> None:
        with self.assertRaisesRegex(PipelineConfigError, "requires obb_center"):
            build_ns_export_command(
                Path("/data/training/config.yml"),
                Path("/data/export/splat.ply"),
                obb_scale=[1.0, 5.0, 1.0],
            )

    def test_viewer_scene_endpoint_maps_to_configured_splat(self) -> None:
        splat_path = Path("/data/export/road_scene.splat")
        handler = object.__new__(SecureRequestHandler)
        handler.server = SimpleNamespace(splat_path=splat_path)

        translated_path = handler.translate_path("/scene.splat?cache=123")

        self.assertEqual(translated_path, str(splat_path))

    def test_publish_to_viewer_creates_link_to_dataset_splat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            tmp_path = Path(temporary_dir)
            splat_path = tmp_path / "dataset" / "export" / "splat.splat"
            splat_path.parent.mkdir(parents=True)
            splat_path.write_bytes(b"test-splat")
            viewer_link = tmp_path / "viewer" / "splat_files" / "current.splat"

            published_path = publish_to_viewer(splat_path, viewer_link)

            self.assertTrue(published_path.is_symlink())
            self.assertEqual(published_path.resolve(), splat_path.resolve())

            # A later export must switch the viewer to the new dataset without
            # copying the model or modifying the previous export.
            next_splat = tmp_path / "next_dataset" / "export" / "splat.splat"
            next_splat.parent.mkdir(parents=True)
            next_splat.write_bytes(b"next-scene")
            publish_to_viewer(next_splat, viewer_link)
            self.assertEqual(viewer_link.read_bytes(), b"next-scene")
            self.assertEqual(viewer_link.resolve(), next_splat.resolve())
            self.assertEqual(splat_path.read_bytes(), b"test-splat")

    def test_viewer_uses_plain_splat_extension(self) -> None:
        index_path = PIPELINE_ROOT / "viewer" / "index.html"
        index_html = index_path.read_text(encoding="utf-8")

        self.assertIn("addSplatScene('splat_files/current.splat'", index_html)
        self.assertNotIn("current.splat?", index_html)


if __name__ == "__main__":
    unittest.main()
