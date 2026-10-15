"""Offline integration checks for paths after reorganizing the project."""

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
from iDriving_pipeline.kafka import producer
from iDriving_pipeline.pipeline_config import PipelinePaths


class ProjectLayoutTests(unittest.TestCase):
    def test_producer_and_pipeline_load_the_same_default_template(self):
        paths = PipelinePaths.from_config(
            {"paths": {"dataset_dir": "/tmp/layout-test"}}, ROOT / "config/config.yaml"
        )
        self.assertEqual(paths.kafka_message_path, producer.DEFAULT_MESSAGE_FILE)
        message = producer.load_json(paths.kafka_message_path)
        self.assertIn("body", message["records"][0])

    def test_pipeline_stage_commands_resolve_to_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            (dataset / "video.mp4").touch()
            (dataset / "images").mkdir()
            for index in range(10):
                (dataset / "images" / f"{index}.jpg").touch()
            (dataset / "transforms.json").write_text(json.dumps({"frames": [{}] * 10}))
            (dataset / "sparse_pc.ply").touch()
            (dataset / "training").mkdir()
            (dataset / "training/config.yml").touch()
            (dataset / "export").mkdir()
            (dataset / "export/splat.ply").touch()
            (dataset / "export/splat.splat").touch()
            config = dataset / "config.yaml"
            config.write_text(
                f"paths:\n  dataset_dir: {dataset}\n"
                f"  video_path: {dataset / 'video.mp4'}\n"
                "  splatfacto_output_dir: training\n"
            )
            with patch.object(run_pipeline, "run_stage") as stages, \
                    patch.object(run_pipeline.subprocess, "run") as viewer, \
                    contextlib.redirect_stdout(io.StringIO()):
                run_pipeline.main([
                    "--config", str(config), "--send_kafka_message", "--start_viewer"
                ])

            expected_scripts = [
                "preprocess/process_drone_data.py", "preprocess/run_hloc.py",
                "train/train_splatfacto.py", "export/export_splat.py", "kafka/producer.py",
            ]
            commands = [call.args[1] for call in stages.call_args_list]
            self.assertEqual([Path(cmd[1]) for cmd in commands],
                             [ROOT / name for name in expected_scripts])
            for command in commands:
                self.assertTrue(Path(command[1]).is_file())
            training_command, export_command = commands[2:4]
            self.assertEqual(
                training_command[training_command.index("--training_config_path") + 1],
                export_command[export_command.index("--training_config_path") + 1],
            )
            message_path = commands[-1][commands[-1].index("--message") + 1]
            producer.load_json(Path(message_path))
            self.assertEqual(Path(viewer.call_args.args[0][1]), ROOT / "viewer/server.py")
            self.assertTrue(Path(viewer.call_args.args[0][1]).is_file())


if __name__ == "__main__":
    unittest.main()
