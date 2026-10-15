"""Check saved training configurations without training or loading a GPU model."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from nerfstudio.configs.experiment_config import ExperimentConfig
from iDriving_pipeline.train.train_splatfacto import build_splatfacto_config, train_splatfacto


class TrainingOutputLayoutTests(unittest.TestCase):
    def test_idriving_config_saved_and_reloaded_with_flat_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            (dataset / "transforms.json").write_text('{"frames": []}')
            output = dataset / "splatfacto_output"
            config = build_splatfacto_config(dataset, output)
            config.set_timestamp()
            self.assertEqual(config.get_base_dir(), output)
            self.assertEqual(config.get_checkpoint_dir(), output / "nerfstudio_models")
            config.save_config()
            saved = yaml.load((output / "config.yml").read_text(), Loader=yaml.Loader)
            self.assertEqual(saved.get_checkpoint_dir(), output / "nerfstudio_models")
            self.assertEqual(list(output.iterdir()), [output / "config.yml"])

    def test_old_configs_retain_nested_layout(self):
        config = ExperimentConfig(
            _target=object,
            output_dir=Path("/tmp/outputs"), experiment_name="scene",
            method_name="splatfacto", timestamp="old-run",
        )
        # Old serialized configs do not contain the new field.
        del config.__dict__["output_dir_is_run_dir"]
        loaded = yaml.load(yaml.dump(config), Loader=yaml.Loader)
        self.assertEqual(loaded.get_base_dir(), Path("/tmp/outputs/scene/splatfacto/old-run"))

    def test_new_training_does_not_overwrite_existing_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            (dataset / "transforms.json").write_text('{"frames": []}')
            output = dataset / "splatfacto_output"
            checkpoints = output / "nerfstudio_models"
            checkpoints.mkdir(parents=True)
            checkpoint = checkpoints / "step-000029999.ckpt"
            checkpoint.write_bytes(b"previous model")
            with patch("iDriving_pipeline.train.train_splatfacto.train_main") as train:
                with self.assertRaisesRegex(FileExistsError, "already contains checkpoints"):
                    train_splatfacto(dataset, output)
                train.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b"previous model")


if __name__ == "__main__":
    unittest.main()
