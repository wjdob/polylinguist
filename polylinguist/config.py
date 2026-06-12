from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path
    cache_dir: Path
    model_artifacts_dir: Path
    settings_file: Path
    installed_models_file: Path
    metadata_cache_file: Path
    generated_subtitles_dir: Path

    @classmethod
    def detect(cls) -> "AppPaths":
        root = Path(os.getenv("POLYLINGUIST_HOME", Path.home() / ".polylinguist"))
        cache_dir = root / "cache"
        generated_subtitles_dir = cache_dir / "subtitles"
        model_artifacts_dir = root / "models"
        return cls(
            root=root,
            cache_dir=cache_dir,
            model_artifacts_dir=model_artifacts_dir,
            settings_file=root / "settings.json",
            installed_models_file=root / "installed_models.json",
            metadata_cache_file=cache_dir / "model_metadata.json",
            generated_subtitles_dir=generated_subtitles_dir,
        )

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.generated_subtitles_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppConfig:
    bind_host: str
    bind_port: int
    public_base_url: str | None
    admin_token: str | None

    @classmethod
    def detect(cls) -> "AppConfig":
        bind_host = (
            os.getenv("POLYLINGUIST_BIND_HOST")
            or os.getenv("POLYLINGUIST_HOST")
            or "127.0.0.1"
        )
        bind_port = int(
            os.getenv("POLYLINGUIST_BIND_PORT")
            or os.getenv("POLYLINGUIST_PORT")
            or "8000"
        )
        public_base_url = _normalize_public_base_url(os.getenv("POLYLINGUIST_PUBLIC_BASE_URL"))
        admin_token = (os.getenv("POLYLINGUIST_ADMIN_TOKEN") or "").strip() or None
        return cls(
            bind_host=bind_host,
            bind_port=bind_port,
            public_base_url=public_base_url,
            admin_token=admin_token,
        )

    @property
    def admin_enabled(self) -> bool:
        return bool(self.admin_token)

    def external_base_url(self, fallback_base_url: str) -> str:
        return self.public_base_url or fallback_base_url.rstrip("/")


def _normalize_public_base_url(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().rstrip("/")
