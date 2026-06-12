from __future__ import annotations

import json
from pathlib import Path


class InstalledModelRegistry:
    def __init__(self, storage_file: Path) -> None:
        self.storage_file = storage_file

    def load(self) -> dict[str, dict[str, object]]:
        if not self.storage_file.exists():
            return {}
        return json.loads(self.storage_file.read_text(encoding="utf-8"))

    def is_installed(self, provider: str, model_id: str, target: str | None = None) -> bool:
        data = self.load()
        if target is not None:
            metadata = data.get(self._key(provider, model_id, target))
            return bool(metadata is not None and self._is_active_entry(metadata))
        metadata = data.get(self._key(provider, model_id))
        if metadata is not None and self._is_active_entry(metadata):
            return True
        prefix = f"{provider}:{model_id}#"
        return any(key.startswith(prefix) and self._is_active_entry(metadata) for key, metadata in data.items())

    def is_removed(self, provider: str, model_id: str, target: str | None = None) -> bool:
        data = self.load()
        key = self._key(provider, model_id, target)
        metadata = data.get(key)
        return bool(metadata is not None and str(metadata.get("status", "")).lower() == "removed")

    def mark_installed(
        self,
        provider: str,
        model_id: str,
        metadata: dict[str, object] | None = None,
        target: str | None = None,
    ) -> None:
        data = self.load()
        payload = dict(metadata or {})
        payload["status"] = "installed"
        data[self._key(provider, model_id, target)] = payload
        self.storage_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def mark_removed(
        self,
        provider: str,
        model_id: str,
        target: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        data = self.load()
        payload = dict(metadata or {})
        payload["status"] = "removed"
        data[self._key(provider, model_id, target)] = payload
        self.storage_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def metadata_for(self, provider: str, model_id: str, target: str | None = None) -> dict[str, object] | None:
        data = self.load()
        if target is not None:
            metadata = data.get(self._key(provider, model_id, target))
            return metadata if metadata is not None and self._is_active_entry(metadata) else None
        metadata = data.get(self._key(provider, model_id))
        return metadata if metadata is not None and self._is_active_entry(metadata) else None

    def installed_targets(self, provider: str, model_id: str) -> set[str]:
        data = self.load()
        prefix = f"{provider}:{model_id}#"
        targets = {
            key.split("#", 1)[1]
            for key, metadata in data.items()
            if key.startswith(prefix) and "#" in key and self._is_active_entry(metadata)
        }
        metadata = data.get(self._key(provider, model_id))
        if metadata is not None and self._is_active_entry(metadata):
            targets.add("cpu")
            targets.add("cuda")
        return targets

    def entries(self) -> dict[str, dict[str, object]]:
        return self.load()

    def active_entries(self) -> dict[str, dict[str, object]]:
        return {
            key: metadata
            for key, metadata in self.load().items()
            if self._is_active_entry(metadata)
        }

    def clear(self) -> None:
        if self.storage_file.exists():
            self.storage_file.unlink()

    @staticmethod
    def _is_active_entry(metadata: dict[str, object]) -> bool:
        return str(metadata.get("status", "installed")).lower() != "removed"

    @staticmethod
    def _key(provider: str, model_id: str, target: str | None = None) -> str:
        if target:
            return f"{provider}:{model_id}#{target}"
        return f"{provider}:{model_id}"
