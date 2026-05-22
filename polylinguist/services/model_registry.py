from __future__ import annotations

import json
from pathlib import Path


class InstalledModelRegistry:
    def __init__(self, storage_file: Path) -> None:
        self.storage_file = storage_file

    def load(self) -> dict[str, dict[str, str]]:
        if not self.storage_file.exists():
            return {}
        return json.loads(self.storage_file.read_text(encoding="utf-8"))

    def is_installed(self, provider: str, model_id: str) -> bool:
        return self._key(provider, model_id) in self.load()

    def mark_installed(self, provider: str, model_id: str, metadata: dict[str, str] | None = None) -> None:
        data = self.load()
        data[self._key(provider, model_id)] = metadata or {}
        self.storage_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def metadata_for(self, provider: str, model_id: str) -> dict[str, str] | None:
        return self.load().get(self._key(provider, model_id))

    @staticmethod
    def _key(provider: str, model_id: str) -> str:
        return f"{provider}:{model_id}"
