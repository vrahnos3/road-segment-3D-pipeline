#!/usr/bin/env python3
"""
Downloads an object from MinIO given bucket/key
"""

import os
import sys
from pathlib import Path

# Configuration
MINIO_ENDPOINT = ""
ACCESS_KEY = ""
SECRET_KEY = ""
DEFAULT_BUCKET = ""
DEFAULT_KEY = "test-image.png"
DOWNLOAD_DIR = str(Path(__file__).resolve().parents[2] / "Downloaded")
CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 30


class DownloadProgress:
    """Adapt MinIO's byte-count callbacks to a Rich progress bar."""

    def __init__(self, progress):
        self.progress = progress
        self.task = progress.add_task("Waiting for MinIO", total=None, start=False)

    def set_meta(self, object_name: str, total_length: int):
        self.progress.update(self.task, description=os.path.basename(object_name), total=total_length)
        self.progress.start_task(self.task)

    def update(self, size: int):
        self.progress.advance(self.task, size)


def download_object(bucket: str, key: str, download_dir: str = DOWNLOAD_DIR) -> str:
    """Download an object by bucket/key to the local download directory."""
    from minio import Minio
    import urllib3
    from rich.progress import (
        BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn,
    )

    print("=" * 60)
    print("MinIO Download (claim-check consumer)")
    print("=" * 60)
    print()

    os.makedirs(download_dir, exist_ok=True)
    destination = os.path.abspath(os.path.join(download_dir, os.path.basename(key)))
    print(f"Bucket: {bucket}")
    print(f"Key: {key}")
    print(f"Destination: {destination}")
    print()

    # The read timeout limits waiting for socket data, not total download time.
    # Disable hidden HTTP retries so a broken connection fails visibly.
    http = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=CONNECT_TIMEOUT_SECONDS, read=READ_TIMEOUT_SECONDS),
        retries=False,
    )
    try:
        # Connect to MinIO
        client = Minio(
            MINIO_ENDPOINT,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=False,
            http_client=http,
        )

        # Validate bucket
        print("Connecting to MinIO and checking the bucket...", flush=True)
        print(
            f"Network timeouts: connect {CONNECT_TIMEOUT_SECONDS}s, "
            f"socket read {READ_TIMEOUT_SECONDS}s.",
            flush=True,
        )
        if not client.bucket_exists(bucket):
            raise FileNotFoundError(f"Bucket '{bucket}' does not exist")

        # Download object
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            DownloadColumn(),
            TransferSpeedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            refresh_per_second=2,
        ) as progress:
            client.fget_object(bucket, key, destination, progress=DownloadProgress(progress))
        size = os.path.getsize(destination)
        print("Download complete.")
        print(f"Saved to: {destination}")
        print(f"Size: {size} bytes")
        return destination

    except urllib3.exceptions.TimeoutError as error:
        raise TimeoutError(
            "MinIO connection or download stalled. Check the VPN and MinIO server, "
            "then retry. Any partial file is incomplete; this downloader restarts "
            "the transfer from the beginning."
        ) from error
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        http.clear()

if __name__ == "__main__":
    bucket = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_BUCKET
    key = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_KEY
    download_object(bucket, key)
