from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from polylinguist.schemas import AddonSettings, SettingsEnvelope


class SettingsStore:
    def __init__(self, settings_file: Path) -> None:
        self.settings_file = settings_file

    def load(self) -> SettingsEnvelope:
        if not self.settings_file.exists():
            return SettingsEnvelope(settings=AddonSettings(), updated_at=None)
        data = json.loads(self.settings_file.read_text(encoding="utf-8"))
        return SettingsEnvelope.model_validate(data)

    def save(self, settings: AddonSettings) -> SettingsEnvelope:
        envelope = SettingsEnvelope(
            settings=settings,
            updated_at=datetime.now(timezone.utc),
        )
        self.settings_file.write_text(
            envelope.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return envelope
