from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from polylinguist.config import AppPaths
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


SUBTITLE_RENDER_VERSION = "3"


@dataclass
class AppServices:
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

        if not self.translation_manager.is_installed(provider, settings.selected_model_id):
            payload = encode_subtitle_payload(
                {
                    "status_only": "true",
                    "status_title": "Polylinguist setup required.",
                    "status_detail": f"Install {provider.title()} model {settings.selected_model_id} from the Polylinguist configurator before using this subtitle.",
                    "status_hint": "Open http://127.0.0.1:8001/configure, install the selected model, then re-select this subtitle in Stremio.",
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

    @staticmethod
    def _display_lang(settings: AddonSettings) -> str:
        if settings.format_mode == "translated_only":
            return settings.target_lang
        return f"{settings.source_lang}+{settings.target_lang}"


def create_services(paths: AppPaths | None = None) -> AppServices:
    app_paths = paths or AppPaths.detect()
    app_paths.ensure()
    settings_store = SettingsStore(app_paths.settings_file)
    model_registry = InstalledModelRegistry(app_paths.installed_models_file)
    return AppServices(
        paths=app_paths,
        settings_store=settings_store,
        system_profile_service=SystemProfileService(),
        model_registry=model_registry,
        model_catalog=ModelCatalogService(app_paths.metadata_cache_file, model_registry),
        translation_manager=TranslationManager(model_registry),
        subtitle_provider=OpenSubtitlesProvider(),
        subtitle_cache=SubtitleCache(app_paths.generated_subtitles_dir),
        install_job_manager=InstallJobManager(),
        subtitle_generation_tracker=SubtitleGenerationTracker(),
    )
