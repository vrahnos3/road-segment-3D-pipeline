"""Read ACCELI upload notifications and download their files from MinIO."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from minIO.download_minio import DEFAULT_BUCKET, download_object


def ready_files(
    value: bytes, *, default_bucket: str = DEFAULT_BUCKET
) -> list[tuple[str, str]]:
    """Read the agreed JSON envelope; ignore messages unrelated to ready uploads."""
    payload = json.loads(value)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("Expected a JSON object containing a records list")
    files = []
    for record in payload["records"]:
        if not isinstance(record, dict):
            raise ValueError("Each record must be an object")
        header, body = record.get("header", {}), record.get("body", {})
        if not isinstance(header, dict) or not isinstance(body, dict):
            raise ValueError("Record header and body must be objects")
        if header.get("producer-id") != "ACCELI" or body.get("file_upload_status") != "ready":
            continue
        bucket = body.get("bucket", default_bucket)
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError("body.bucket must be a non-empty string")
        key = body.get("file_name")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("A ready ACCELI upload must contain a non-empty body.file_name")
        if key.rsplit("/", 1)[-1] in ("", ".", ".."):
            raise ValueError("body.file_name must identify a file, not a directory")
        files.append((bucket, key))
    return files


async def process_message(message, args) -> None:
    # Validate the entire envelope before downloading any of its files.
    uploads = ready_files(message.value, default_bucket=args.bucket)
    for index, (bucket, key) in enumerate(uploads):
        # Kafka coordinates keep equal filenames and separate uploads isolated.
        job_dir = args.work_dir / f"{message.topic}-{message.partition}-{message.offset}-{index}"
        print(f"Processing MinIO object {bucket}/{key}", flush=True)
        downloaded_path = await asyncio.to_thread(
            download_object, bucket, key, str(job_dir)
        )
        print(f"Download complete: {downloaded_path}", flush=True)
        print(f"Dataset directory: {job_dir}", flush=True)


async def consume(consumer, args) -> None:
    try:
        await consumer.start()
        print(f"Waiting for ACCELI uploads on {args.topic}...", flush=True)
        async for message in consumer:
            # Any validation/download failure exits without committing.
            # Restart with the same group ID to retry the failed Kafka message.
            await process_message(message, args)
            await consumer.commit()
    finally:
        await consumer.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True, help="Incoming upload topic agreed with ACCELI.")
    parser.add_argument("--broker", default="157.90.19.188:9092")
    parser.add_argument("--group_id", default="certh-3d-smart-video-consumer")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET,
                        help="Fallback bucket when body.bucket is absent from the message.")
    parser.add_argument("--work_dir", type=Path, default=PIPELINE_ROOT / "dataset",
                        help="Parent folder for datasets (default: iDriving_pipeline/dataset).")
    parser.add_argument("--max_poll_interval_ms", type=int, default=86_400_000,
                        help="Must exceed download time for one complete message (default: 24 hours).")
    args = parser.parse_args()
    args.work_dir = args.work_dir.expanduser().resolve()
    if args.max_poll_interval_ms <= 0:
        parser.error("--max_poll_interval_ms must be positive")

    from aiokafka import AIOKafkaConsumer

    async def run():
        consumer = AIOKafkaConsumer(
            args.topic,
            bootstrap_servers=args.broker,
            group_id=args.group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            max_poll_interval_ms=args.max_poll_interval_ms,
        )
        await consume(consumer, args)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Consumer stopped.")


if __name__ == "__main__":
    main()
