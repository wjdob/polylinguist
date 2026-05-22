from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path
    cache_dir: Path
    settings_file: Path
    installed_models_file: Path
    metadata_cache_file: Path
    generated_subtitles_dir: Path

    @classmethod
    def detect(cls) -> "AppPaths":
        root = Path(os.getenv("POLYLINGUIST_HOME", Path.home() / ".polylinguist"))
        cache_dir = root / "cache"
        generated_subtitles_dir = cache_dir / "subtitles"
        return cls(
            root=root,
            cache_dir=cache_dir,
            settings_file=root / "settings.json",
            installed_models_file=root / "installed_models.json",
            metadata_cache_file=cache_dir / "model_metadata.json",
            generated_subtitles_dir=generated_subtitles_dir,
        )

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.generated_subtitles_dir.mkdir(parents=True, exist_ok=True)
