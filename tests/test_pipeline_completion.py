"""Registration and training-completion checks without GPU or broker access."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from iDriving_pipeline import run_pipeline
from iDriving_pipeline.pipeline_config import publish_training_config


class CompletionTests(unittest.TestCase):
    def test_two_registered_images_stop_pipeline_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            images = dataset / "images"
            images.mkdir()
            for index in range(103):
                (images / f"{index}.jpg").touch()
            (dataset / "transforms.json").write_text(json.dumps({"frames": [{}, {}]}))
            (dataset / "sparse_pc.ply").touch()
            config = dataset / "config.yaml"
            config.write_text(f"paths:\n  dataset_dir: {dataset}\n")
            with patch.object(run_pipeline, "run_stage") as stage, \
                    patch.object(run_pipeline.subprocess, "run") as viewer, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "2/103"):
                    run_pipeline.main(["--config", str(config), "--skip_image_extraction"])
            self.assertEqual(stage.call_count, 1)
            self.assertIn("run_hloc.py", stage.call_args.args[1][1])
            viewer.assert_not_called()

    def test_registration_fraction_is_enforced_and_configurable(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            for index in range(30):
                (dataset / f"{index}.png").touch()
            transforms = dataset / "transforms.json"
            transforms.write_text(json.dumps({"frames": [{}] * 10}))
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Insufficient reconstruction"):
                    run_pipeline.check_reconstruction(transforms, dataset, 10, 0.5)
                run_pipeline.check_reconstruction(transforms, dataset, 10, 0.3)

    def test_completed_run_config_is_copied_without_changing_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            actual = output / "experiment/splatfacto/timestamp/config.yml"
            checkpoint_dir = actual.parent / "nerfstudio_models"
            checkpoint_dir.mkdir(parents=True)
            actual.write_text("timestamp: timestamp\nexperiment_name: experiment\n")
            (checkpoint_dir / "step-000029999.ckpt").write_bytes(b"checkpoint")
            destination = output / "config.yml"
            publish_training_config(actual, destination, checkpoint_dir, 29999)
            self.assertEqual(destination.read_bytes(), actual.read_bytes())
            self.assertTrue((checkpoint_dir / "step-000029999.ckpt").is_file())

    def test_interrupted_training_cannot_publish_a_config_from_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "run/config.yml"
            source.parent.mkdir()
            source.write_text("new run")
            checkpoint_dir = source.parent / "nerfstudio_models"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "step-000002000.ckpt").write_bytes(b"partial training")
            destination = output / "config.yml"
            destination.write_text("previous completed run")
            with self.assertRaisesRegex(RuntimeError, "Training did not complete"):
                publish_training_config(source, destination, checkpoint_dir, 29999)
            self.assertEqual(destination.read_text(), "previous completed run")


if __name__ == "__main__":
    unittest.main()
