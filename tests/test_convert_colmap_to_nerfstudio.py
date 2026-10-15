import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from preprocess.convert_colmap_to_nerfstudio import (  # noqa: E402
    build_parser,
    convert_reconstruction,
    find_model_dir,
)


class ExistingColmapTests(unittest.TestCase):
    def test_find_model_in_common_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for relative_dir, extension in (("colmap/sparse/0", ".bin"), ("text_model", ".txt")):
                with self.subTest(relative_dir=relative_dir):
                    model_dir = root / relative_dir
                    model_dir.mkdir(parents=True)
                    for name in ("cameras", "images", "points3D"):
                        (model_dir / f"{name}{extension}").touch()
                    search_dir = root / "colmap" if extension == ".bin" else model_dir
                    self.assertEqual(find_model_dir(search_dir), (model_dir, extension))

    def test_incomplete_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_dir = Path(temporary_dir)
            (model_dir / "cameras.bin").touch()
            with self.assertRaisesRegex(FileNotFoundError, "No complete COLMAP model"):
                find_model_dir(model_dir)

    def test_existing_output_is_not_overwritten_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            for name in ("cameras", "images", "points3D"):
                (model_dir / f"{name}.bin").touch()
            images_dir = root / "images"
            images_dir.mkdir()
            output_dir = root / "dataset"
            output_dir.mkdir()
            transforms_path = output_dir / "transforms.json"
            transforms_path.write_text("original", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                convert_reconstruction(model_dir, images_dir, output_dir)
            self.assertEqual(transforms_path.read_text(encoding="utf-8"), "original")

    def test_cli_accepts_underscore_paths(self) -> None:
        args = build_parser().parse_args(
            ["--recon_dir", "/data/model", "--image_dir", "/data/images", "--output_dir", "/data/dataset"]
        )
        self.assertEqual(args.recon_dir, Path("/data/model"))
        self.assertEqual(args.output_dir, Path("/data/dataset"))

    def test_binary_and_text_models_export_matching_cameras_and_points(self) -> None:
        # These integration checks use the same installed dependencies as training.
        # The remaining tests still run in a lightweight Python environment.
        try:
            import numpy as np
            from PIL import Image as PILImage

            from nerfstudio.data.utils.colmap_parsing_utils import Camera, Image, Point3D, write_model
            from nerfstudio.process_data import colmap_utils  # noqa: F401
        except ImportError as error:
            self.skipTest(f"Run in the nerfstudio environment for integration coverage: {error}")

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            image_dir = root / "vggt_images"
            image_dir.mkdir()
            cameras = {
                camera_id: Camera(camera_id, "PINHOLE", 8, 6, np.array([focal, focal, 4.0, 3.0]))
                for camera_id, focal in ((7, 5.0), (9, 6.0))
            }
            images = {}
            for image_id, camera_id in ((1, 7), (2, 9)):
                name = f"nested/frame_{image_id}.png"
                (image_dir / name).parent.mkdir(exist_ok=True)
                PILImage.new("RGB", (8, 6), color=(20, 40, 60)).save(image_dir / name)
                images[image_id] = Image(
                    image_id,
                    np.array([1.0, 0.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 0.0]),
                    camera_id,
                    name,
                    np.array([[1.0, 1.0]]),
                    np.array([1]),
                )
            points = {
                1: Point3D(
                    1, np.array([1.0, 2.0, 3.0]), np.array([20, 40, 60]), 0.1,
                    np.array([1, 2]), np.array([0, 0]),
                )
            }

            for extension in (".bin", ".txt"):
                with self.subTest(extension=extension):
                    model_dir = root / extension[1:]
                    model_dir.mkdir()
                    write_model(cameras, images, points, model_dir, ext=extension)
                    output_dir = root / f"dataset_{extension[1:]}"
                    with contextlib.redirect_stdout(io.StringIO()):
                        count = convert_reconstruction(model_dir, image_dir, output_dir)
                    self.assertEqual(count, 2)
                    metadata = json.loads((output_dir / "transforms.json").read_text(encoding="utf-8"))
                    self.assertEqual(metadata["ply_file_path"], "sparse_pc.ply")
                    for frame in metadata["frames"]:
                        image_id = frame["colmap_im_id"]
                        self.assertEqual((output_dir / frame["file_path"]).resolve(), image_dir / images[image_id].name)
                        self.assertEqual(frame["fl_x"], 5.0 if image_id == 1 else 6.0)
                        expected_pose = np.eye(4)
                        expected_pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
                        np.testing.assert_allclose(frame["transform_matrix"], expected_pose)
                    ply_data = (output_dir / "sparse_pc.ply").read_text(encoding="utf-8")
                    self.assertIn("element vertex 1", ply_data)
                    np.testing.assert_allclose([float(value) for value in ply_data.splitlines()[-1].split()], [1, 3, -2, 20, 40, 60])

                    PILImage.new("RGB", (4, 3)).save(image_dir / images[1].name)
                    with self.assertRaisesRegex(ValueError, "Image size mismatch"):
                        convert_reconstruction(model_dir, image_dir, output_dir, overwrite=True)
                    # Failed validation preserves the previously exported dataset.
                    self.assertEqual((output_dir / "sparse_pc.ply").read_text(encoding="utf-8"), ply_data)
                    PILImage.new("RGB", (8, 6)).save(image_dir / images[1].name)


if __name__ == "__main__":
    unittest.main()
