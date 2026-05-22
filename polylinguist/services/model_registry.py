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

    def is_installed(self, provider: str, model_id: str, target: str | None = None) -> bool:
        data = self.load()
        if target is not None:
            return self._key(provider, model_id, target) in data
        if self._key(provider, model_id) in data:
            return True
        prefix = f"{provider}:{model_id}#"
        return any(key.startswith(prefix) for key in data)

    def mark_installed(
        self,
        provider: str,
        model_id: str,
        metadata: dict[str, str] | None = None,
        target: str | None = None,
    ) -> None:
        data = self.load()
        data[self._key(provider, model_id, target)] = metadata or {}
        self.storage_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def metadata_for(self, provider: str, model_id: str, target: str | None = None) -> dict[str, str] | None:
        data = self.load()
        if target is not None:
            return data.get(self._key(provider, model_id, target))
        return data.get(self._key(provider, model_id))

    def installed_targets(self, provider: str, model_id: str) -> set[str]:
        data = self.load()
        prefix = f"{provider}:{model_id}#"
        targets = {
            key.split("#", 1)[1]
            for key in data
            if key.startswith(prefix) and "#" in key
        }
        if self._key(provider, model_id) in data:
            targets.add("cpu")
            targets.add("cuda")
        return targets

    @staticmethod
    def _key(provider: str, model_id: str, target: str | None = None) -> str:
        if target:
            return f"{provider}:{model_id}#{target}"
        return f"{provider}:{model_id}"
