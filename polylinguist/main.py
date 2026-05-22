from __future__ import annotations

import os
import socket

import uvicorn
from polylinguist.config import AppConfig


def main() -> None:
    config = AppConfig.detect()
    host = config.bind_host
    requested_port = config.bind_port
    port = _find_available_port(host, requested_port)
    print(f"Starting Polylinguist on http://{host}:{port}/configure")
    uvicorn.run("polylinguist.app:create_app", factory=True, host=host, port=port, reload=False)


def _find_available_port(host: str, requested_port: int, attempts: int = 20) -> int:
    for port in range(requested_port, requested_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"Polylinguist could not find a free port starting at {host}:{requested_port}."
    )


if __name__ == "__main__":
    main()
