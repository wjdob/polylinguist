from __future__ import annotations

import argparse
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a Caddy reverse-proxy config for Polylinguist.")
    parser.add_argument("--public-base-url", required=True, help="Public HTTPS base URL, for example https://subs.example.net")
    parser.add_argument("--bind-host", default="127.0.0.1", help="Polylinguist bind host")
    parser.add_argument("--bind-port", type=int, default=8000, help="Polylinguist bind port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed = urlparse(args.public_base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise SystemExit("The public base URL must be an HTTPS URL with a host name.")
    upstream = f"http://{args.bind_host}:{args.bind_port}"
    print(
        f"""{parsed.netloc} {{
    encode zstd gzip
    reverse_proxy {upstream}
}}"""
    )


if __name__ == "__main__":
    main()
