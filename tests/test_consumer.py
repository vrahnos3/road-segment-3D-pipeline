"""Offline checks for upload filtering, download destinations, and offset handling."""

import json
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from iDriving_pipeline.kafka import consumer


def message(producer="ACCELI", status="ready", filename="videos/road.mp4"):
    return SimpleNamespace(topic="uploads", partition=0, offset=12, value=json.dumps({
        "records": [{"header": {"producer-id": producer},
                     "body": {"file_upload_status": status, "file_name": filename}}]
    }).encode())


class ConsumerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = SimpleNamespace(topic="uploads", bucket="uc2.1-certh-3d-smart",
                                    work_dir=Path("/tmp/jobs"))

    def test_filter_and_validate(self):
        self.assertEqual(consumer.ready_files(message().value),
                         [(self.args.bucket, "videos/road.mp4")])
        self.assertEqual(consumer.ready_files(message(producer="CERTH").value), [])
        self.assertEqual(consumer.ready_files(message(status="uploading").value), [])
        for value in (b"invalid json", b"null", b'{"records": [null]}',
                      message(filename=None).value):
            with self.subTest(value=value), self.assertRaises(ValueError):
                consumer.ready_files(value)

    async def test_download_goes_directly_in_dataset_without_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            self.args.work_dir = Path(directory)
            dataset_dir = Path(directory) / "uploads-0-12-0"

            def download_file(bucket, key, destination):
                output = Path(destination) / Path(key).name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"test-video")
                return str(output)

            with patch.object(consumer, "download_object", side_effect=download_file) as download, \
                    patch.object(consumer.asyncio, "create_subprocess_exec") as run:
                await consumer.process_message(message(), self.args)
            download.assert_called_once_with(self.args.bucket, "videos/road.mp4", str(dataset_dir))
            self.assertEqual(list(dataset_dir.iterdir()), [dataset_dir / "road.mp4"])
            run.assert_not_called()

    async def test_message_bucket_overrides_fallback_for_each_record(self):
        notification = message()
        payload = json.loads(notification.value)
        payload["records"][0]["body"]["bucket"] = "partner-upload-bucket"
        payload["records"].append(json.loads(message(filename="second.mp4").value)["records"][0])
        notification.value = json.dumps(payload).encode()
        with patch.object(consumer, "download_object", return_value="/tmp/video.mp4") as download:
            await consumer.process_message(notification, self.args)
        self.assertEqual([call.args[:2] for call in download.call_args_list],
                         [("partner-upload-bucket", "videos/road.mp4"),
                          (self.args.bucket, "second.mp4")])
        self.assertEqual([call.args[2] for call in download.call_args_list],
                         ["/tmp/jobs/uploads-0-12-0", "/tmp/jobs/uploads-0-12-1"])

    def test_invalid_bucket_is_rejected(self):
        for bucket in (None, "", " ", 123):
            payload = json.loads(message().value)
            payload["records"][0]["body"]["bucket"] = bucket
            with self.subTest(bucket=bucket), self.assertRaisesRegex(ValueError, "body.bucket"):
                consumer.ready_files(json.dumps(payload).encode())

    async def test_download_failure_leaves_message_uncommitted(self):
        client = AsyncMock()
        client.__aiter__.return_value = [message()]
        with patch.object(consumer, "download_object", side_effect=OSError("download failed")):
            with self.assertRaisesRegex(OSError, "download failed"):
                await consumer.consume(client, self.args)
        client.commit.assert_not_awaited()
        client.stop.assert_awaited_once()

    async def test_json_download_commits_without_starting_pipeline(self):
        client = AsyncMock()
        client.__aiter__.return_value = [message(filename="tests/small.json")]
        with patch.object(consumer, "download_object", return_value="/tmp/small.json") as download, \
                patch.object(consumer.asyncio, "create_subprocess_exec") as run:
            await consumer.consume(client, self.args)
        download.assert_called_once_with(self.args.bucket, "tests/small.json",
                                         "/tmp/jobs/uploads-0-12-0")
        run.assert_not_called()
        client.commit.assert_awaited_once()
        client.stop.assert_awaited_once()

    async def test_all_records_validated_before_any_download(self):
        notification = message()
        payload = json.loads(notification.value)
        payload["records"].append(json.loads(message(filename=None).value)["records"][0])
        notification.value = json.dumps(payload).encode()
        with patch.object(consumer, "download_object") as download:
            with self.assertRaises(ValueError):
                await consumer.process_message(notification, self.args)
        download.assert_not_called()

    def test_download_requires_filename(self):
        for filename in (None, "", " ", "tests/", ".", "tests/.."):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                consumer.ready_files(message(filename=filename).value)

    async def test_commit_only_after_success_and_stop_on_failure(self):
        for failure in (None, RuntimeError("download failed"), ValueError("invalid message")):
            with self.subTest(failure=failure):
                client = AsyncMock()
                client.__aiter__.return_value = [message(), message()]
                with patch.object(consumer, "process_message", side_effect=failure) as process:
                    if failure:
                        with self.assertRaises(type(failure)):
                            await consumer.consume(client, self.args)
                        client.commit.assert_not_awaited()
                        self.assertEqual(process.await_count, 1)
                    else:
                        await consumer.consume(client, self.args)
                        self.assertEqual(client.commit.await_count, 2)
                client.stop.assert_awaited_once()

if __name__ == "__main__":
    unittest.main()
