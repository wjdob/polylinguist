from __future__ import annotations

import hashlib
from pathlib import Path


class SubtitleCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def key_for_text(self, seed: str) -> str:
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    def path_for_key(self, cache_key: str) -> Path:
        safe_key = self.key_for_text(cache_key)
        return self.root / f"{safe_key}.srt"

    def get(self, cache_key: str) -> str | None:
        path = self.path_for_key(cache_key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()

    def put(self, cache_key: str, content: str) -> Path:
        path = self.path_for_key(cache_key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        return path
