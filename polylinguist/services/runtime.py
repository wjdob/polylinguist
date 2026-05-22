from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import shutil

from polylinguist.config import AppConfig, AppPaths
from polylinguist.schemas import AddonSettings
from polylinguist.services.cache import SubtitleCache
from polylinguist.services.install_jobs import InstallJobManager
from polylinguist.services.model_catalog import ModelCatalogService
from polylinguist.services.model_registry import InstalledModelRegistry
from polylinguist.services.settings_store import SettingsStore
from polylinguist.services.subtitle_sources import OpenSubtitlesProvider, SubtitleSourceExtra
from polylinguist.services.subtitles import (
    SubtitleCandidate,
    decode_subtitle_payload,
    encode_subtitle_payload,
    parse_subtitle_text,
    render_dual_srt,
)
from polylinguist.services.subtitle_jobs import SubtitleGenerationTracker
from polylinguist.services.system_profile import SystemProfileService
from polylinguist.services.translation import TranslationManager, TranslationRequest
from polylinguist.services.local_models import local_model_artifact_dir


SUBTITLE_RENDER_VERSION = "3"


@dataclass
class AppServices:
    config: AppConfig
    paths: AppPaths
    settings_store: SettingsStore
    system_profile_service: SystemProfileService
    model_registry: InstalledModelRegistry
    model_catalog: ModelCatalogService
    translation_manager: TranslationManager
    subtitle_provider: OpenSubtitlesProvider
    subtitle_cache: SubtitleCache
    install_job_manager: InstallJobManager
    subtitle_generation_tracker: SubtitleGenerationTracker
    _subtitle_generation_tasks: dict[str, asyncio.Task[str]] = field(default_factory=dict)

    async def build_subtitle_options(
        self,
        settings: AddonSettings,
        media_type: str,
        media_id: str,
        extra: SubtitleSourceExtra,
        base_url: str,
        limit: int = 3,
    ) -> list[dict[str, str]]:
        provider = settings.preferred_provider
        if provider == "auto":
            catalog = self.model_catalog.list_models(
                settings.source_lang,
                settings.target_lang,
                self.system_profile_service.detect(),
            )
            provider = catalog.recommended_provider or "marian"
            if catalog.recommended_model_id:
                settings = settings.model_copy(update={"selected_model_id": catalog.recommended_model_id})

        if not settings.selected_model_id:
            return []

        profile = self.system_profile_service.detect()
        allowed, reason = self.model_catalog.validate_processing_target(
            profile,
            provider,
            settings.selected_model_id,
            settings.source_lang,
            settings.target_lang,
            settings.processing_device,
        )
        if not allowed:
            payload = encode_subtitle_payload(
                {
                    "status_only": "true",
                    "status_title": "Polylinguist configuration needs attention.",
                    "status_detail": reason or "The selected processing target is not supported on this machine.",
                    "status_hint": f"Open {self.config.external_base_url(base_url)}/configure and choose a supported processing target.",
                }
            )
            return [
                {
                    "id": f"poly-invalid-{provider}",
                    "lang": self._display_lang(settings),
                    "label": f"Polylinguist configuration issue - {settings.source_lang.upper()}+{settings.target_lang.upper()}",
                    "url": f"{base_url}/subs/{payload}.srt",
                }
            ]

        if not self.translation_manager.is_installed(provider, settings.selected_model_id, settings.processing_device):
            payload = encode_subtitle_payload(
                {
                    "status_only": "true",
                    "status_title": "Polylinguist setup required.",
                    "status_detail": f"Install {provider.title()} model {settings.selected_model_id} from the Polylinguist configurator before using this subtitle.",
                    "status_hint": f"Open {self.config.external_base_url(base_url)}/configure, install the selected model, then re-select this subtitle in Stremio.",
                }
            )
            return [
                {
                    "id": f"poly-setup-{provider}",
                    "lang": self._display_lang(settings),
                    "label": f"Polylinguist setup required - {settings.source_lang.upper()}+{settings.target_lang.upper()} - {provider.title()}",
                    "url": f"{base_url}/subs/{payload}.srt",
                }
            ]

        candidates = await self.subtitle_provider.list_candidates(
            media_type,
            media_id,
            settings.source_lang,
            extra,
            limit=limit,
        )
        items: list[dict[str, str]] = []
        for index, candidate in enumerate(candidates, start=1):
            payload = encode_subtitle_payload(
                {
                    "provider": provider,
                    "model_id": settings.selected_model_id,
                    "source_lang": settings.source_lang,
                    "target_lang": settings.target_lang,
                    "processing_device": settings.processing_device,
                    "format_mode": settings.format_mode,
                    "render_version": SUBTITLE_RENDER_VERSION,
                    "candidate_id": candidate.subtitle_id,
                    "candidate_url": candidate.url,
                    "candidate_lang": candidate.lang,
                    "candidate_format": candidate.format,
                    "candidate_label": candidate.match_label,
                    "candidate_name": candidate.label,
                    "media_filename": extra.filename or "",
                    "configure_url": f"Open {self.config.external_base_url(base_url)}/configure for progress, then re-select this subtitle when it is ready.",
                }
            )
            if index == 1:
                self.ensure_subtitle_generation(payload)
            items.append(
                {
                    "id": f"poly-{candidate.subtitle_id}-{index}",
                    "lang": self._display_lang(settings),
                    "label": f"Dual {settings.source_lang.upper()}+{settings.target_lang.upper()} - {provider.title()} - {candidate.match_label} #{index}",
                    "url": f"{base_url}/subs/{payload}.srt",
                }
            )
        return items

    async def generate_translated_subtitle(self, cache_key: str) -> str:
        cached = self.subtitle_cache.get(cache_key)
        if cached is not None:
            self.subtitle_generation_tracker.completed(cache_key, "Served from cache.")
            return cached

        self.subtitle_generation_tracker.progress(cache_key, "starting", "Preparing subtitle generation.")
        try:
            payload = decode_subtitle_payload(cache_key)
            settings = AddonSettings(
                source_lang=payload["source_lang"],
                target_lang=payload["target_lang"],
                preferred_provider=payload["provider"],
                selected_model_id=payload["model_id"],
                processing_device=payload.get("processing_device", "auto"),
                format_mode=payload["format_mode"],
            )
            candidate = SubtitleCandidate(
                subtitle_id=payload["candidate_id"],
                url=payload["candidate_url"],
                lang=payload["candidate_lang"],
                label=payload.get("candidate_name", payload.get("candidate_label", payload["candidate_lang"])),
                source="opensubtitles",
                score=0.0,
                format=payload.get("candidate_format", "srt"),
                match_label=payload.get("candidate_label", "match"),
            )
            self.subtitle_generation_tracker.describe(
                cache_key,
                source_subtitle_name=candidate.label,
                source_subtitle_id=candidate.subtitle_id,
                source_subtitle_url=candidate.url,
                media_filename=str(payload.get("media_filename") or ""),
            )
            self.subtitle_generation_tracker.progress(cache_key, "fetch", "Downloading primary subtitle.")
            text = await self.subtitle_provider.fetch_subtitle_text(candidate)
            self.subtitle_generation_tracker.progress(cache_key, "parse", "Parsing subtitle cues.")
            cues = parse_subtitle_text(text)
            if not cues:
                raise ValueError("Primary subtitle could not be parsed into cues.")

            def progress(stage: str, message: str) -> None:
                self.subtitle_generation_tracker.progress(cache_key, stage, message)

            progress("translate", f"Translating {len(cues)} subtitle cues.")
            translations = await asyncio.to_thread(
                self.translation_manager.translate_batch,
                TranslationRequest(
                    provider=payload["provider"],
                    model_id=payload["model_id"],
                    source_lang=payload["source_lang"],
                    target_lang=payload["target_lang"],
                    device_preference=payload.get("processing_device", "auto"),
                ),
                [cue.text for cue in cues],
                progress,
            )
            self.subtitle_generation_tracker.progress(cache_key, "render", "Rendering translated SRT.")
            rendered = render_dual_srt(cues, translations, settings)
            self.subtitle_cache.put(cache_key, rendered)
            self.subtitle_generation_tracker.completed(cache_key, f"Rendered {len(cues)} cues.")
            return rendered
        except Exception as exc:
            self.subtitle_generation_tracker.failed(cache_key, exc)
            raise

    def ensure_subtitle_generation(self, cache_key: str) -> asyncio.Task[str]:
        task = self._subtitle_generation_tasks.get(cache_key)
        if task is not None and not task.done():
            self.subtitle_generation_tracker.progress(
                cache_key,
                "queued",
                "Reusing in-flight subtitle generation for this selection.",
            )
            return task
        task = asyncio.create_task(self.generate_translated_subtitle(cache_key))
        self._subtitle_generation_tasks[cache_key] = task
        return task

    def remove_model_installation(self, provider: str, model_id: str, processing_device: str) -> dict[str, object]:
        normalized = (processing_device or "auto").lower()
        removed_paths: list[str] = []
        notes: list[str] = []

        self.translation_manager.forget_model(provider, model_id, normalized)

        if provider == "marian" and normalized in {"directml", "openvino_gpu"}:
            artifact_dir = local_model_artifact_dir(self.paths.model_artifacts_dir, provider, model_id, normalized)
            if artifact_dir.exists():
                self._remove_tree(artifact_dir)
                removed_paths.append(str(artifact_dir))
            self.model_registry.mark_removed(provider, model_id, target=normalized)
            detail = f"Removed Polylinguist-managed Marian {normalized} artifacts for {model_id}."
        elif provider == "marian":
            self.model_registry.mark_removed(provider, model_id)
            detail = f"Removed {model_id} from Polylinguist's Marian install list for CPU/CUDA use."
            notes.append("Shared Hugging Face model cache was left in place because it may be used by other tools on this machine.")
        elif provider == "nllb":
            self.model_registry.mark_removed(provider, model_id)
            self.translation_manager.forget_model(provider, model_id, "cpu")
            self.translation_manager.forget_model(provider, model_id, "cuda")
            detail = f"Removed {model_id} from Polylinguist's NLLB install list."
            notes.append("Shared Hugging Face model cache was left in place because it may be used by other tools on this machine.")
        elif provider == "argos":
            self.model_registry.mark_removed(provider, model_id)
            detail = f"Removed {model_id} from Polylinguist's Argos install list."
            notes.append("Shared Argos package data was left in place to avoid deleting translations used outside Polylinguist.")
        else:
            self.model_registry.mark_removed(provider, model_id, target=normalized if normalized != "auto" else None)
            detail = f"Removed {provider}:{model_id} from Polylinguist's install list."

        self._clear_selected_model_if_matches(provider, model_id, normalized)
        self._reset_subtitle_runtime_state()
        self.subtitle_cache.clear()
        return {
            "detail": detail,
            "removed_paths": removed_paths,
            "notes": notes,
        }

    def uninstall_local_data(self) -> dict[str, object]:
        removed_paths: list[str] = []
        notes = [
            "Shared Hugging Face and Argos caches were left in place to avoid deleting data used by other tools on this machine.",
            "Polylinguist's Python package or executable is still installed. Stop the service and uninstall that runtime separately if you want a full system-level uninstall.",
        ]

        self._reset_subtitle_runtime_state()
        self.translation_manager.clear_runtime_state()
        self.model_registry.clear()

        for path in [self.paths.cache_dir, self.paths.model_artifacts_dir]:
            if path.exists():
                self._remove_tree(path)
                removed_paths.append(str(path))

        for file_path in [self.paths.settings_file, self.paths.installed_models_file, self.paths.metadata_cache_file]:
            if file_path.exists():
                file_path.unlink()
                removed_paths.append(str(file_path))

        self.paths.ensure()
        return {
            "detail": "Removed Polylinguist local data and reset the service to first-run defaults.",
            "removed_paths": removed_paths,
            "notes": notes,
        }

    @staticmethod
    def _display_lang(settings: AddonSettings) -> str:
        if settings.format_mode == "translated_only":
            return settings.target_lang
        return f"{settings.source_lang}+{settings.target_lang}"

    def _clear_selected_model_if_matches(self, provider: str, model_id: str, normalized_target: str) -> None:
        envelope = self.settings_store.load()
        settings = envelope.settings
        if settings.preferred_provider != provider or settings.selected_model_id != model_id:
            return
        if normalized_target in {"auto", "cpu", "cuda"}:
            if settings.processing_device not in {"auto", "cpu", "cuda"}:
                return
        elif settings.processing_device != normalized_target:
            return
        self.settings_store.save(
            settings.model_copy(
                update={
                    "preferred_provider": "auto",
                    "selected_model_id": None,
                }
            )
        )

    def _reset_subtitle_runtime_state(self) -> None:
        for cache_key, task in list(self._subtitle_generation_tasks.items()):
            if not task.done():
                task.cancel()
            self._subtitle_generation_tasks.pop(cache_key, None)
        self.subtitle_generation_tracker.clear()

    @staticmethod
    def _remove_tree(path: Path) -> None:
        shutil.rmtree(path)


def create_services(paths: AppPaths | None = None, config: AppConfig | None = None) -> AppServices:
    app_paths = paths or AppPaths.detect()
    app_paths.ensure()
    app_config = config or AppConfig.detect()
    settings_store = SettingsStore(app_paths.settings_file)
    model_registry = InstalledModelRegistry(app_paths.installed_models_file)
    return AppServices(
        config=app_config,
        paths=app_paths,
        settings_store=settings_store,
        system_profile_service=SystemProfileService(),
        model_registry=model_registry,
        model_catalog=ModelCatalogService(app_paths.metadata_cache_file, model_registry, app_paths.model_artifacts_dir),
        translation_manager=TranslationManager(model_registry, app_paths.model_artifacts_dir),
        subtitle_provider=OpenSubtitlesProvider(),
        subtitle_cache=SubtitleCache(app_paths.generated_subtitles_dir),
        install_job_manager=InstallJobManager(),
        subtitle_generation_tracker=SubtitleGenerationTracker(),
    )
