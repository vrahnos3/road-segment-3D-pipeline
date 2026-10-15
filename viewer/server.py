"""Serve the web viewer and the SPLAT selected by the pipeline configuration."""

from __future__ import annotations

import argparse
from functools import partial
import http.server
from pathlib import Path
import socketserver
import ssl
import subprocess
import sys
from typing import Optional, cast
from urllib.parse import urlsplit


BASE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = BASE_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PipelineConfigError,
    PipelinePaths,
    load_pipeline_config,
)


def get_vpn_ip() -> Optional[str]:
    try:
        result = subprocess.check_output(["ip", "addr", "show", "tun0"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None

    for line in result.splitlines():
        if "inet " in line:
            return line.strip().split(" ")[1].split("/")[0]
    return None


class SecureTCPServer(socketserver.TCPServer):
    allow_reuse_address = True
    splat_path: Path

    def handle_error(self, request, client_address) -> None:  # type: ignore[no-untyped-def]
        error = sys.exc_info()[1]
        if isinstance(error, ssl.SSLError):
            return
        super().handle_error(request, client_address)


class SecureRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve static viewer files and map /scene.splat to the configured output."""

    server: SecureTCPServer

    def translate_path(self, path: str) -> str:
        if urlsplit(path).path == "/scene.splat":
            return str(self.server.splat_path)
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


def serve(splat_path: Path, port: int) -> None:
    splat_path = splat_path.expanduser().resolve()
    if not splat_path.is_file():
        raise FileNotFoundError(f"SPLAT file not found: {splat_path}")

    vpn_ip = get_vpn_ip()
    print("_" * 45)
    print("\n3DGS Splat Viewer Server")
    print("_" * 45)
    print(f"\nScene:  {splat_path}")
    print(f"Local:  https://localhost:{port}")
    if vpn_ip:
        print(f"VPN:    https://{vpn_ip}:{port}")
    else:
        print("VPN:    not connected (tun0 not found)")
    print("_" * 45)
    print("\nPress Ctrl+C to stop the server")
    print("_" * 45, flush=True)

    handler = partial(SecureRequestHandler, directory=str(BASE_DIR))
    with SecureTCPServer(("", port), handler) as raw_server:
        server = cast(SecureTCPServer, raw_server)
        server.splat_path = splat_path
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(BASE_DIR / "cert.pem", BASE_DIR / "key.pem")
        server.socket = context.wrap_socket(server.socket, server_side=True)
        server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the configured iDriving SPLAT scene.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset_dir", type=Path)
    parser.add_argument("--splat_path", type=Path, help="Override paths.splat_output_path.")
    parser.add_argument("--port", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_pipeline_config(config_path)
    paths = PipelinePaths.from_config(
        config,
        config_path,
        {"dataset_dir": args.dataset_dir, "splat_output_path": args.splat_path},
    )
    viewer_config = config.get("viewer", {}) or {}
    if not isinstance(viewer_config, dict):
        raise PipelineConfigError("Configuration section 'viewer' must be a mapping.")
    port = args.port if args.port is not None else viewer_config.get("port", 8000)
    serve(paths.splat_output_path, port)


if __name__ == "__main__":
    try:
        main()
    except PipelineConfigError as error:
        raise SystemExit(f"Configuration error: {error}") from error
