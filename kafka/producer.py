#!/usr/bin/env python3
"""
Simple iDriving Kafka producer.

The JSON file contains the complete message that will be sent to Kafka.
Only correlation-id and message-timestamp are generated at send time.
"""

import argparse
import asyncio
import copy
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from aiokafka import AIOKafkaProducer

MESSAGE_VISUALIZATION_LINK = "http://157.90.19.188:8081/"
KAFKA_BROKER = "157.90.19.188:9092"
TOPIC_NAME = "idriving_certh_3d_smart_uc2.1"
CLIENT_ID = "CERTH"

DEFAULT_MESSAGE_FILE = Path(__file__).resolve().parent / "templates" / "file_upload_ready.json"


def load_json(json_file: Path) -> Dict[str, Any]:
    with json_file.open("r", encoding="utf-8") as file:
        message = json.load(file)

    validate_message(message, json_file)
    return message


def validate_message(message: Dict[str, Any], message_file: Path) -> None:
    try:
        record = message["records"][0]
        header = record["header"]
        record["body"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            f"{message_file} must contain records[0].header and records[0].body"
        ) from error

    if not isinstance(header, dict):
        raise ValueError(f"{message_file} records[0].header must be a JSON object")


def prepare_message(message_template: Dict[str, Any]) -> Dict[str, Any]:
    message = copy.deepcopy(message_template)
    header = message["records"][0]["header"]

    header["correlation-id"] = str(uuid.uuid4())
    header["message-timestamp"] = datetime.now().isoformat()

    return message


def create_kafka_headers(message: Dict[str, Any]) -> List[Tuple[str, bytes]]:
    """Takes every field from the JSON header and sends it also as a Kafka record header"""
    json_header = message["records"][0]["header"]

    kafka_headers = []
    for name, value in json_header.items():
        if value is not None:
            kafka_headers.append((name, str(value).encode("utf-8")))

    return kafka_headers


class IDrivingKafkaProducer:
    def __init__(self) -> None:
        """Initialize the Kafka producer with proper configuration"""
        self.producer: Any | None = None

    async def connect(self) -> None:
        """Connect to the Kafka broker"""
        
        print(f"🔗 Connecting to Kafka broker: {KAFKA_BROKER}")
        print(f"See messages in AKHQ Management UI: {MESSAGE_VISUALIZATION_LINK}")
        self.producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            acks="all",
            client_id=CLIENT_ID,
        )
        await self.producer.start()
        print("✅ Connected to Kafka broker successfully!")

    async def send_message(self, message: Dict[str, Any], key: Optional[str] = None) -> bool:
        if self.producer is None:
            print("❌ Kafka producer is not connected.")
            return False

        kafka_headers = create_kafka_headers(message)
        message_value = json.dumps(message).encode("utf-8")
        message_key = key.encode("utf-8") if key else None

        print("\n" + "=" * 80)
        print("📤 Sending message to Kafka")
        print("=" * 80)
        print(f"Topic: {TOPIC_NAME}")
        print(f"Key: {key if key else 'null'}")

        print("\nKafka record headers:")
        print("-" * 80)
        for name, value in kafka_headers:
            print(f"{name:<22} | {value.decode('utf-8')}")

        print("\nJSON message:")
        print("-" * 80)
        print(json.dumps(message, indent=2))

        try:
            metadata = await self.producer.send_and_wait(
                topic=TOPIC_NAME,
                key=message_key,
                value=message_value,
                headers=kafka_headers,
            )
        except Exception as error:
            print(f"Error sending message: {error}")
            return False

        kafka_timestamp = None
        if metadata.timestamp is not None:
            kafka_timestamp = datetime.fromtimestamp(metadata.timestamp / 1000)

        print("\n✅ Message sent successfully.")
        print(f"Topic: {metadata.topic}")
        print(f"Partition: {metadata.partition}")
        print(f"Offset: {metadata.offset}")
        print(f"Kafka timestamp: {kafka_timestamp}")
        print("=" * 80)

        return True

    async def close(self) -> None:
        """Close the Kafka producer"""
        if self.producer is not None:
            await self.producer.stop()
            print("🔌 Kafka producer connection closed.")


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Send an iDriving Kafka message.")
    parser.add_argument(
        "--message",
        type=Path,
        default=DEFAULT_MESSAGE_FILE,
        help=f"Path to the complete JSON message template (default: {DEFAULT_MESSAGE_FILE})",
    )
    parser.add_argument("--key", help="Optional Kafka message key")
    args = parser.parse_args()

    try:
        message_template = load_json(args.message)
        message = prepare_message(message_template)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Could not load message template: {error}")
        return 1

    producer = IDrivingKafkaProducer()

    try:
        await producer.connect()
        success = await producer.send_message(message=message, key=args.key)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by the user.")
        return 130
    except Exception as error:
        print(f"❌ Unexpected error: {error}")
        return 1
    finally:
        await producer.close()


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
