from __future__ import annotations

import asyncio
import base64
import html
import json
from typing import Any
from urllib.parse import parse_qsl, quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse

from polylinguist.config import AppConfig, AppPaths
from polylinguist.schemas import (
    AddonSettings,
    CleanupResponse,
    InstallJobResponse,
    LanguageResponse,
    ModelCatalogResponse,
    ModelInstallRequest,
    ModelRemoveRequest,
    SettingsEnvelope,
)
from polylinguist.services.languages import list_languages
from polylinguist.services.model_catalog import ModelDescriptor
from polylinguist.services.runtime import AppServices, create_services
from polylinguist.services.subtitle_sources import SubtitleSourceExtra
from polylinguist.services.subtitles import decode_subtitle_payload
from polylinguist.services.translation import TranslationError


def create_app(services: AppServices | None = None) -> FastAPI:
    app = FastAPI(title="Polylinguist", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.services = services or create_services(AppPaths.detect(), AppConfig.detect())

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> str:
        services = get_services(request)
        return render_configure_page(
            str(request.base_url).rstrip("/"),
            services.config.external_base_url(str(request.base_url).rstrip("/")),
            services.config.admin_enabled,
        )

    @app.get("/configure", response_class=HTMLResponse)
    async def configure(request: Request) -> str:
        services = get_services(request)
        return render_configure_page(
            str(request.base_url).rstrip("/"),
            services.config.external_base_url(str(request.base_url).rstrip("/")),
            services.config.admin_enabled,
        )

    @app.get("/api/languages", response_model=list[LanguageResponse])
    async def get_languages(request: Request) -> list[LanguageResponse]:
        require_admin_access(request)
        return [
            LanguageResponse(
                code=item.canonical,
                label=item.label,
                iso639_1=item.iso639_1,
                nllb=item.nllb_code,
            )
            for item in list_languages()
        ]

    @app.get("/api/system/profile")
    async def get_system_profile(request: Request) -> dict[str, Any]:
        require_admin_access(request)
        services = get_services(request)
        return services.system_profile_service.detect().to_response().model_dump()

    @app.get("/api/settings", response_model=SettingsEnvelope)
    async def get_settings(request: Request) -> SettingsEnvelope:
        require_admin_access(request)
        return get_services(request).settings_store.load()

    @app.put("/api/settings", response_model=SettingsEnvelope)
    async def put_settings(request: Request, settings: AddonSettings) -> SettingsEnvelope:
        require_admin_access(request)
        services = get_services(request)
        if settings.selected_model_id and settings.preferred_provider != "auto":
            allowed, reason = services.model_catalog.validate_processing_target(
                services.system_profile_service.detect(),
                settings.preferred_provider,
                settings.selected_model_id,
                settings.source_lang,
                settings.target_lang,
                settings.processing_device,
            )
            if not allowed:
                raise HTTPException(status_code=400, detail=reason or "Unsupported processing target.")
        return services.settings_store.save(settings)

    @app.get("/api/models", response_model=ModelCatalogResponse)
    async def get_models(
        request: Request,
        source_lang: str = Query(...),
        target_lang: str = Query(...),
    ) -> ModelCatalogResponse:
        require_admin_access(request)
        services = get_services(request)
        return services.model_catalog.list_models(
            source_lang,
            target_lang,
            services.system_profile_service.detect(),
        )

    @app.post("/api/models/install", response_model=InstallJobResponse)
    async def install_model(request: Request, payload: ModelInstallRequest) -> InstallJobResponse:
        require_admin_access(request)
        services = get_services(request)
        profile = services.system_profile_service.detect()
        catalog = services.model_catalog.list_models(
            payload.source_lang,
            payload.target_lang,
            profile,
        )
        option = next(
            (
                ModelDescriptor(
                    provider=item.provider,
                    model_id=item.model_id,
                    label=item.label,
                    source_lang=item.source_lang,
                    target_lang=item.target_lang,
                    size_mb=item.size_mb,
                    available=item.available,
                    direct=item.direct,
                    installed=item.installed,
                    installed_targets=tuple(item.installed_targets),
                    supported_targets=tuple(item.supported_targets),
                    recommended_target=item.recommended_target,
                    availability_reason=item.availability_reason,
                    note=item.note,
                    license=item.license,
                    install_strategy=item.install_strategy,
                )
                for item in catalog.models
                if item.provider == payload.provider and item.model_id == payload.model_id
            ),
            None,
        )
        if option is None or not option.available:
            raise HTTPException(status_code=404, detail="Requested model is not available for this pair.")
        requested_target = payload.processing_device
        effective_target = (option.recommended_target or "cpu") if requested_target == "auto" else requested_target
        allowed, reason = services.model_catalog.validate_processing_target(
            profile,
            payload.provider,
            payload.model_id,
            payload.source_lang,
            payload.target_lang,
            effective_target,
        )
        if not allowed:
            raise HTTPException(status_code=400, detail=reason or "Unsupported processing target.")
        if payload.persist_selection:
            current = services.settings_store.load().settings
            services.settings_store.save(
                current.model_copy(
                    update={
                        "source_lang": payload.source_lang,
                        "target_lang": payload.target_lang,
                        "preferred_provider": payload.provider,
                        "selected_model_id": payload.model_id,
                        "processing_device": payload.processing_device,
                    }
                )
            )
        job = services.install_job_manager.create_job(
            provider=payload.provider,
            model_id=payload.model_id,
            install_fn=lambda progress: services.translation_manager.install(
                option,
                progress,
                effective_target,
            ),
        )
        return InstallJobResponse.model_validate(job.to_dict())

    @app.get("/api/models/install/{job_id}", response_model=InstallJobResponse)
    async def get_install_job(request: Request, job_id: str) -> InstallJobResponse:
        require_admin_access(request)
        services = get_services(request)
        job = services.install_job_manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Install job not found.")
        return InstallJobResponse.model_validate(job.to_dict())

    @app.post("/api/models/remove", response_model=CleanupResponse)
    async def remove_model(request: Request, payload: ModelRemoveRequest) -> CleanupResponse:
        require_admin_access(request)
        services = get_services(request)
        result = services.remove_model_installation(
            payload.provider,
            payload.model_id,
            payload.processing_device,
        )
        return CleanupResponse.model_validate(result)

    @app.get("/api/subtitles/status")
    async def get_subtitle_activity(request: Request) -> dict[str, Any]:
        require_admin_access(request)
        services = get_services(request)
        return {
            "jobs": [job.to_dict() for job in services.subtitle_generation_tracker.recent()],
        }

    @app.get("/api/subtitles/status/{cache_key}")
    async def get_subtitle_status(request: Request, cache_key: str) -> dict[str, Any]:
        require_admin_access(request)
        services = get_services(request)
        job = services.subtitle_generation_tracker.get(cache_key)
        if job is not None:
            return job.to_dict()
        if services.subtitle_cache.get(cache_key) is not None:
            return {
                "cache_key": cache_key,
                "status": "completed",
                "stage": "completed",
                "message": "Subtitle is ready in cache.",
                "detail": "Served from disk cache.",
                "log_lines": ["[completed] Subtitle is ready in cache."],
                "updated_at": None,
            }
        return {
            "cache_key": cache_key,
            "status": "unknown",
            "stage": "unknown",
            "message": "No generation activity has been recorded for this subtitle yet.",
            "detail": None,
            "log_lines": [],
            "updated_at": None,
        }

    @app.get("/manifest.json")
    async def manifest(request: Request) -> dict[str, Any]:
        services = get_services(request)
        settings = services.settings_store.load().settings
        base_url = services.config.external_base_url(str(request.base_url).rstrip("/"))
        config_token = encode_config(settings)
        return build_manifest(base_url, settings, config_token)

    @app.get("/{config_token}/manifest.json")
    async def configured_manifest(request: Request, config_token: str) -> dict[str, Any]:
        services = get_services(request)
        settings = decode_config(config_token)
        base_url = services.config.external_base_url(str(request.base_url).rstrip("/"))
        return build_manifest(base_url, settings, config_token)

    @app.get("/{config_token}/subtitles/{media_type}/{media_id}/{extra_path:path}")
    async def subtitles_with_extra_path(
        request: Request,
        config_token: str,
        media_type: str,
        media_id: str,
        extra_path: str,
    ) -> dict[str, Any]:
        if not extra_path.endswith(".json"):
            raise HTTPException(status_code=404, detail="Subtitle route must end with .json.")
        return await build_subtitles_response(
            request=request,
            config_token=config_token,
            media_type=media_type,
            media_id=media_id,
            path_extra=extra_path.removesuffix(".json"),
        )

    @app.get("/{config_token}/subtitles/{media_type}/{media_id}.json")
    async def subtitles_without_extra_path(
        request: Request,
        config_token: str,
        media_type: str,
        media_id: str,
        videoHash: str | None = None,
        videoSize: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        return await build_subtitles_response(
            request=request,
            config_token=config_token,
            media_type=media_type,
            media_id=media_id,
            filename=filename,
            videoSize=videoSize,
            videoHash=videoHash,
        )

    @app.get("/subs/{cache_key}.srt", response_class=PlainTextResponse)
    async def generated_subtitle(request: Request, cache_key: str) -> PlainTextResponse:
        services = get_services(request)
        try:
            payload = decode_subtitle_payload(cache_key)
        except Exception:
            payload = {}
        if payload.get("status_only") == "true":
            return PlainTextResponse(
                render_status_subtitle(
                    str(payload.get("status_title") or "Polylinguist status."),
                    str(payload.get("status_detail") or ""),
                    str(payload.get("status_hint") or ""),
                ),
                media_type="text/srt; charset=utf-8",
                headers={"Cache-Control": "no-store"},
            )
        cached = services.subtitle_cache.get(cache_key)
        if cached is not None:
            services.subtitle_generation_tracker.completed(cache_key, "Served from cache.")
            return PlainTextResponse(
                cached,
                media_type="text/srt; charset=utf-8",
                headers={"Cache-Control": "no-store"},
            )

        task = services.ensure_subtitle_generation(cache_key)
        try:
            content = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=subtitle_wait_timeout_seconds(payload),
            )
        except TimeoutError:
            job = services.subtitle_generation_tracker.get(cache_key)
            message = job.message if job else "Translation is still running."
            return PlainTextResponse(
                render_status_subtitle(
                    "Polylinguist is still translating this subtitle.",
                    message,
                    str(payload.get("configure_url") or "Open the Polylinguist configure page for progress, then re-select this subtitle when it is ready."),
                ),
                media_type="text/srt; charset=utf-8",
                headers={"Cache-Control": "no-store"},
            )
        except TranslationError as exc:
            return PlainTextResponse(
                render_status_subtitle("Polylinguist subtitle generation failed.", str(exc)),
                media_type="text/srt; charset=utf-8",
                headers={"Cache-Control": "no-store"},
            )
        except Exception as exc:
            return PlainTextResponse(
                render_status_subtitle("Polylinguist subtitle generation failed.", str(exc)),
                media_type="text/srt; charset=utf-8",
                headers={"Cache-Control": "no-store"},
            )
        return PlainTextResponse(
            content,
            media_type="text/srt; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    return app


def get_services(request: Request) -> AppServices:
    return request.app.state.services


def get_app_config(request: Request) -> AppConfig:
    return get_services(request).config


def require_admin_access(request: Request) -> None:
    config = get_app_config(request)
    if not config.admin_enabled:
        return
    token = (request.headers.get("X-Polylinguist-Admin-Token") or "").strip()
    if token and token == config.admin_token:
        return
    raise HTTPException(status_code=401, detail="Admin token is required.")


def encode_config(settings: AddonSettings) -> str:
    raw = settings.model_dump_json(exclude_none=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_config(config_token: str) -> AddonSettings:
    padding = "=" * (-len(config_token) % 4)
    data = base64.urlsafe_b64decode(config_token + padding)
    return AddonSettings.model_validate(json.loads(data.decode("utf-8")))


async def build_subtitles_response(
    request: Request,
    config_token: str,
    media_type: str,
    media_id: str,
    path_extra: str | None = None,
    videoHash: str | None = None,
    videoSize: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    services = get_services(request)
    settings = decode_config(config_token)
    extra_params = parse_extra_args(path_extra)
    base_url = services.config.external_base_url(str(request.base_url).rstrip("/"))
    items = await services.build_subtitle_options(
        settings=settings,
        media_type=media_type,
        media_id=media_id,
        extra=SubtitleSourceExtra(
            filename=filename or extra_params.get("filename"),
            video_size=videoSize or extra_params.get("videoSize"),
            video_hash=videoHash or extra_params.get("videoHash"),
        ),
        base_url=base_url,
    )
    return {
        "subtitles": items,
        "cacheMaxAge": 60,
    }


def parse_extra_args(path_extra: str | None) -> dict[str, str]:
    if not path_extra:
        return {}
    return dict(parse_qsl(path_extra, keep_blank_values=True))


def render_status_subtitle(title: str, detail: str, hint: str | None = None) -> str:
    lines = [
        "1",
        "00:00:00,000 --> 99:59:59,000",
        html.escape(title, quote=False),
    ]
    if detail:
        lines.append(html.escape(detail, quote=False))
    if hint:
        lines.append(html.escape(hint, quote=False))
    return "\r\n".join(lines) + "\r\n"


def subtitle_wait_timeout_seconds(payload: dict[str, Any]) -> float:
    provider = str(payload.get("provider") or "")
    device = str(payload.get("processing_device") or "auto").lower()
    if device in {"cuda", "directml", "openvino_gpu"}:
        return 120.0
    if provider == "marian":
        return 240.0
    if provider == "argos":
        return 180.0
    if provider == "nllb":
        return 240.0
    return 180.0


def build_manifest(base_url: str, settings: AddonSettings, config_token: str) -> dict[str, Any]:
    return {
        "id": "local.polylinguist",
        "version": "0.1.0",
        "name": f"Polylinguist ({settings.source_lang.upper()}->{settings.target_lang.upper()})",
        "description": "Local AI subtitle translator sidecar for Stremio.",
        "resources": ["subtitles"],
        "types": ["movie", "series"],
        "idPrefixes": ["tt"],
        "catalogs": [],
        "behaviorHints": {
            "configurable": True,
            "configurationRequired": False,
        },
        "logo": f"{base_url}/configure",
        "background": f"{base_url}/configure",
        "contactEmail": "local@polylinguist.invalid",
        "subtitles": [],
        "stremioAddonsConfig": {
            "issuer": base_url,
        },
        "config": [
            {"key": "source_lang", "type": "text", "title": "Source subtitle language"},
            {"key": "target_lang", "type": "text", "title": "Target translation language"},
            {"key": "provider", "type": "text", "title": "Preferred model provider"},
            {"key": "model_id", "type": "text", "title": "Target model id"},
            {"key": "processing_device", "type": "text", "title": "Processing target"},
        ],
        "links": {
            "configure": f"{base_url}/configure",
            "manifest": f"{base_url}/{config_token}/manifest.json",
        },
    }


def render_configure_page(api_base_url: str, manifest_base_url: str, admin_required: bool) -> str:
    escaped_api_url = quote(api_base_url, safe=":/")
    escaped_manifest_url = quote(manifest_base_url, safe=":/")
    admin_required_literal = "true" if admin_required else "false"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Polylinguist</title>
  <style>
    :root {{
      --bg: #141414;
      --surface: #181818;
      --surface-2: #202020;
      --surface-3: #262626;
      --line: #303030;
      --line-strong: #3f3f3f;
      --ink: #f5f5f1;
      --muted: #b4b4b0;
      --accent: #e50914;
      --accent-strong: #ff2230;
      --accent-soft: #3a1618;
      --success: #46d369;
      --warn: #ffb347;
      --ui: "Inter", "Segoe UI", "Aptos", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--ui);
    }}
    .shell {{
      width: 100%;
      padding: 24px 28px 40px;
    }}
    .hero {{
      display: grid;
      gap: 28px;
      grid-template-columns: minmax(420px, 1.2fr) minmax(540px, 0.95fr);
      align-items: start;
      padding: 20px 0 28px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 24px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: clamp(2.6rem, 7vw, 5.6rem);
      line-height: 0.92;
      letter-spacing: 0;
      font-weight: 800;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 18px;
      color: #ffffff;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .eyebrow::before {{
      content: "";
      width: 34px;
      height: 3px;
      background: var(--accent);
    }}
    .hero-side {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      align-items: stretch;
    }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      padding: 20px;
    }}
    .hero-side .panel {{
      height: 100%;
      display: flex;
      flex-direction: column;
      justify-content: flex-start;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #ffffff;
    }}
    label {{
      display: block;
      margin-bottom: 14px;
      font-size: 0.9rem;
      color: var(--muted);
    }}
    select, input, button {{
      width: 100%;
      margin-top: 6px;
      border: 1px solid var(--line-strong);
      background: var(--surface-2);
      color: var(--ink);
      padding: 12px 13px;
      min-height: 46px;
      font: inherit;
    }}
    select:focus, input:focus {{
      outline: 1px solid var(--accent);
      border-color: var(--accent);
    }}
    button {{
      cursor: pointer;
      background: var(--accent);
      color: #fff;
      border: 1px solid var(--accent);
      transition: transform 160ms ease, opacity 160ms ease;
      font-weight: 700;
    }}
    button:hover {{
      transform: translateY(-1px);
      background: var(--accent-strong);
      border-color: var(--accent-strong);
    }}
    button:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
      transform: none;
      background: var(--surface-3);
      border-color: var(--line);
    }}
    button.secondary {{
      background: transparent;
      color: var(--ink);
      border: 1px solid var(--line-strong);
    }}
    button.danger {{
      background: transparent;
      color: #ffb5bb;
      border: 1px solid #7c1d24;
    }}
    button.danger:hover {{
      background: #301012;
      border-color: #a3232d;
      color: #fff;
    }}
    .model-list {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    .model {{
      border: 1px solid var(--line);
      padding: 16px;
      background: var(--surface-2);
    }}
    .model strong {{ display: block; margin-bottom: 6px; }}
    .meta {{
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.5;
    }}
    .recommended {{
      background: #1f1617;
      border-color: #7c1d24;
      box-shadow: inset 0 0 0 1px rgba(229, 9, 20, 0.16);
    }}
    .install-row {{
      display: flex;
      gap: 10px;
      margin-top: 12px;
    }}
    .install-row button {{ flex: 1; }}
    .status {{
      min-height: 1.4rem;
      color: var(--warn);
      font-size: 0.95rem;
    }}
    .job-status {{
      border: 1px solid var(--line);
      background: var(--surface-2);
      padding: 14px;
      margin-top: 12px;
    }}
    .job-stage {{
      font-weight: 600;
      margin-bottom: 6px;
    }}
    .job-log {{
      margin-top: 10px;
      max-height: 220px;
      overflow: auto;
      padding: 12px;
      background: #111111;
      border: 1px solid var(--line);
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.85rem;
      white-space: pre-wrap;
      color: #d8d8d5;
    }}
    .progress-shell {{
      margin-top: 10px;
    }}
    .progress-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 0.9rem;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .progress-bar {{
      width: 100%;
      height: 10px;
      border: 1px solid var(--line);
      background: #0f0f0f;
      overflow: hidden;
    }}
    .progress-fill {{
      height: 100%;
      background: var(--accent);
      width: 0%;
      transition: width 160ms ease;
    }}
    .manifest {{
      display: grid;
      gap: 10px;
    }}
    .manifest-note {{
      font-size: 0.9rem;
      color: var(--muted);
      line-height: 1.45;
    }}
    .actions-panel {{
      gap: 14px;
    }}
    .actions-panel button {{
      margin-top: 0;
    }}
    .spacer {{
      flex: 1;
    }}
    .workspace {{
      display: grid;
      gap: 24px;
      grid-template-columns: minmax(320px, 380px) minmax(0, 1fr);
      align-items: start;
    }}
    .stack {{
      display: grid;
      gap: 24px;
    }}
    .headline-row {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 12px;
    }}
    .headline-row .meta {{
      max-width: 40rem;
    }}
    .dominant-panel {{
      min-height: calc(100svh - 220px);
    }}
    .activity-list {{
      display: grid;
      gap: 14px;
    }}
    .activity-list .job-log {{
      max-height: 320px;
    }}
    a {{
      color: #ffffff;
    }}
    a:hover {{
      color: #ffffff;
    }}
    @media (max-width: 980px) {{
      .shell {{ padding: 20px 16px 32px; }}
      .hero,
      .hero-side,
      .workspace {{
        grid-template-columns: 1fr;
      }}
      .dominant-panel {{
        min-height: auto;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <div>
        <div class="eyebrow">Stremio Desktop Local Addon • Translation control room</div>
        <h1>Polylinguist</h1>
      </div>
      <div class="hero-side">
        <div class="panel">
          <h2>System Profile</h2>
          <div id="system-profile" class="meta">Detecting local hardware...</div>
        </div>
        <div class="panel">
          <h2>Current Config</h2>
          <label>Target model
            <select id="model-select">
              <option value="auto">Auto recommend</option>
            </select>
          </label>
          <div class="spacer"></div>
          <div id="settings-status" class="status"></div>
        </div>
        <div class="panel actions-panel">
          <h2>Actions</h2>
          <button id="save-settings" type="button" class="secondary">Save local defaults</button>
          <div class="manifest" id="manifest-link">
            <button id="copy-manifest" type="button">Copy manifest URL</button>
            <div id="manifest-copy-status" class="manifest-note"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="workspace">
      <div class="stack">
        <section class="panel">
          <h2>Languages</h2>
          <label>Primary subtitle language
            <select id="source-lang"></select>
          </label>
          <label>Target translation language
            <select id="target-lang"></select>
          </label>
          <label>Subtitle mode
            <select id="format-mode">
              <option value="dual">Dual</option>
              <option value="translated_only">Translated only</option>
            </select>
          </label>
          <label>Processing target
            <select id="processing-device">
              <option value="auto">Auto</option>
              <option value="cpu">CPU</option>
              <option value="cuda">GPU (CUDA)</option>
              <option value="directml">GPU (DirectML)</option>
              <option value="openvino_gpu">GPU (OpenVINO)</option>
            </select>
          </label>
          <button id="refresh-models" type="button">Evaluate models</button>
        </section>

        <section class="panel">
          <div class="headline-row">
            <div>
              <h2>Model Options</h2>
              <div class="meta">Choose the translation backend that matches this machine and the active language pair.</div>
            </div>
          </div>
          <div id="model-status" class="status"></div>
          <div id="job-status" class="job-status" hidden>
            <div id="job-stage" class="job-stage"></div>
            <div id="job-message" class="meta"></div>
            <div id="job-log" class="job-log"></div>
          </div>
          <div id="models" class="model-list"></div>
        </section>
      </div>

      <div class="stack">
        <section class="panel dominant-panel">
          <div class="headline-row">
            <div>
              <h2>Translation Activity</h2>
              <div class="meta">Live generation status for the most recent subtitle requests.</div>
            </div>
          </div>
          <div id="subtitle-activity-status" class="status">No subtitle generation activity yet.</div>
          <div id="subtitle-activity" class="activity-list"></div>
        </section>
      </div>
    </div>
  </div>

  <script>
    const apiBase = "{escaped_api_url}";
    const manifestBase = "{escaped_manifest_url}";
    const adminRequired = {admin_required_literal};
    const ADMIN_TOKEN_STORAGE_KEY = "polylinguistAdminToken";
    const ADMIN_TOKEN_HEADER = "X-Polylinguist-Admin-Token";
    let currentSettings = null;
    let activeInstallJobId = null;
    let activeInstallPoll = null;
    let availableModels = [];
    let latestCatalogPayload = null;
    let systemProfile = null;

    async function loadJson(path, options) {{
      const init = options ? {{ ...options }} : {{}};
      init.headers = new Headers(init.headers || {{}});
      if (adminRequired) {{
        let token = sessionStorage.getItem(ADMIN_TOKEN_STORAGE_KEY) || "";
        if (!token) {{
          token = window.prompt("Enter the Polylinguist admin token to manage this server.") || "";
          if (!token) {{
            throw new Error("Admin token is required.");
          }}
          sessionStorage.setItem(ADMIN_TOKEN_STORAGE_KEY, token);
        }}
        init.headers.set(ADMIN_TOKEN_HEADER, token);
      }}
      let response = await fetch(apiBase + path, init);
      if (response.status === 401 && adminRequired) {{
        sessionStorage.removeItem(ADMIN_TOKEN_STORAGE_KEY);
        const retryToken = window.prompt("Admin token was rejected. Enter the Polylinguist admin token again.") || "";
        if (!retryToken) {{
          throw new Error("Admin token is required.");
        }}
        sessionStorage.setItem(ADMIN_TOKEN_STORAGE_KEY, retryToken);
        init.headers.set(ADMIN_TOKEN_HEADER, retryToken);
        response = await fetch(apiBase + path, init);
      }}
      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json") ? await response.json() : await response.text();
      if (!response.ok) {{
        throw new Error(payload.detail || payload || "Request failed");
      }}
      return payload;
    }}

    function manifestUrl(settings) {{
      const encoded = btoa(JSON.stringify(settings)).replace(/\\+/g, "-").replace(/\\//g, "_").replace(/=+$/g, "");
      return manifestBase + "/" + encoded + "/manifest.json";
    }}

    function fillLanguages(items) {{
      const source = document.getElementById("source-lang");
      const target = document.getElementById("target-lang");
      source.innerHTML = "";
      target.innerHTML = "";
      for (const item of items) {{
        const optionA = document.createElement("option");
        optionA.value = item.code;
        optionA.textContent = item.label + " [" + item.code + "]";
        source.appendChild(optionA);
        const optionB = optionA.cloneNode(true);
        target.appendChild(optionB);
      }}
    }}

    function readSettings() {{
      const selection = document.getElementById("model-select").value;
      let preferredProvider = "auto";
      let selectedModelId = null;
      if (selection !== "auto") {{
        const [provider, modelId] = selection.split("::", 2);
        preferredProvider = provider;
        selectedModelId = modelId;
      }}
      return {{
        source_lang: document.getElementById("source-lang").value,
        target_lang: document.getElementById("target-lang").value,
        preferred_provider: preferredProvider,
        selected_model_id: selectedModelId,
        processing_device: document.getElementById("processing-device").value,
        format_mode: document.getElementById("format-mode").value,
      }};
    }}

    function selectedModelValue(settings) {{
      if (!settings?.selected_model_id || settings?.preferred_provider === "auto") {{
        return "auto";
      }}
      return settings.preferred_provider + "::" + settings.selected_model_id;
    }}

    function renderManifest(settings) {{
      const link = manifestUrl(settings);
      const button = document.getElementById("copy-manifest");
      if (button) {{
        button.dataset.url = link;
      }}
    }}

    async function copyManifestUrl() {{
      const button = document.getElementById("copy-manifest");
      const status = document.getElementById("manifest-copy-status");
      const url = button?.dataset.url || "";
      if (!url) {{
        status.textContent = "Manifest URL is not ready yet.";
        return;
      }}
      try {{
        if (navigator.clipboard?.writeText) {{
          await navigator.clipboard.writeText(url);
        }} else {{
          const probe = document.createElement("textarea");
          probe.value = url;
          document.body.appendChild(probe);
          probe.select();
          document.execCommand("copy");
          document.body.removeChild(probe);
        }}
        status.textContent = "Manifest URL copied.";
      }} catch (error) {{
        status.textContent = "Copy failed. Use the browser address bar fallback.";
      }}
    }}

    function renderSystemProfile(profile) {{
      systemProfile = profile;
      const acceleratorSummary = (profile.accelerators || []).length
        ? " • accelerators: " + profile.accelerators.map((item) => item.name + " [" + item.supported_targets.join(", ") + "]").join(" | ")
        : "";
      document.getElementById("system-profile").textContent =
        profile.os + " / " + profile.arch +
        " • " + profile.cpu_cores + " cores" +
        " • " + profile.total_ram_gb.toFixed(1) + " GB RAM" +
        " • " + profile.free_disk_gb.toFixed(1) + " GB free" +
        " • tier: " + profile.tier +
        " • CUDA: " + (profile.has_cuda ? "yes" : "no") +
        " • MPS: " + (profile.has_mps ? "yes" : "no") +
        acceleratorSummary;
      const supportedTargets = supportedTargetsFromProfile(profile);
      for (const value of ["cuda", "directml", "openvino_gpu"]) {{
        const option = document.querySelector('#processing-device option[value="' + value + '"]');
        if (!option) {{
          continue;
        }}
        option.disabled = !supportedTargets.has(value);
      }}
      if (!supportedTargets.has(document.getElementById("processing-device").value) && document.getElementById("processing-device").value !== "auto") {{
        document.getElementById("processing-device").value = "auto";
      }}
    }}

    function supportedTargetsFromProfile(profile) {{
      const supportedTargets = new Set(["cpu", "auto"]);
      if (profile.has_cuda) {{
        supportedTargets.add("cuda");
      }}
      for (const accelerator of profile.accelerators || []) {{
        for (const target of accelerator.supported_targets || []) {{
          supportedTargets.add(target);
        }}
      }}
      return supportedTargets;
    }}

    function applySettingsToForm(settings) {{
      currentSettings = settings;
      document.getElementById("source-lang").value = currentSettings.source_lang;
      document.getElementById("target-lang").value = currentSettings.target_lang;
      document.getElementById("format-mode").value = currentSettings.format_mode;
      const requestedTarget = currentSettings.processing_device || "auto";
      const supportedTargets = systemProfile ? supportedTargetsFromProfile(systemProfile) : new Set(["auto", "cpu"]);
      document.getElementById("processing-device").value =
        supportedTargets.has(requestedTarget)
          ? requestedTarget
          : "auto";
      renderManifest(currentSettings);
    }}

    function renderModels(payload) {{
      latestCatalogPayload = payload;
      availableModels = payload.models || [];
      const container = document.getElementById("models");
      container.innerHTML = "";
      renderModelSelect(payload);
      if (!payload.models.length) {{
        container.textContent = "No models available for this language pair yet.";
        return;
      }}
      for (const item of payload.models) {{
        const div = document.createElement("div");
        div.className = "model" + (item.recommended ? " recommended" : "");
        const note = item.note ? "<div class=\\"meta\\">" + item.note + "</div>" : "";
        const availability = item.availability_reason ? "<div class=\\"meta\\">" + item.availability_reason + "</div>" : "";
        const selectedTarget = effectiveProcessingTarget(item);
        const targetList = (item.supported_targets || []).join(", ");
        const targetMeta = "<div class=\\"meta\\">Targets: " + targetList + " • Recommended: " + (item.recommended_target || "cpu") + "</div>";
        const canInstall = isModelSelectableForTarget(item, selectedTarget);
        const installedForTarget = (item.installed_targets || []).includes(selectedTarget);
        const badge = item.recommended ? "Recommended for this machine" : item.available ? "Available" : "Unavailable";
        const removeButton = installedForTarget
          ? '<button type="button" class="secondary danger" data-action="remove" data-provider="' + item.provider + '" data-model-id="' + item.model_id + '" data-target="' + selectedTarget + '">Remove ' + selectedTarget + '</button>'
          : "";
        div.innerHTML =
          "<strong>" + item.label + "</strong>" +
          "<div class=\\"meta\\">" + badge + " • " + item.size_mb + " MB • " + item.license + "</div>" +
          targetMeta +
          availability +
          note +
          "<div class=\\"install-row\\">" +
            "<button type=\\"button\\" " + (canInstall ? "" : "disabled") + " data-provider=\\"" + item.provider + "\\" data-model-id=\\"" + item.model_id + "\\" data-target=\\"" + selectedTarget + "\\">" +
              (installedForTarget ? "Reinstall for " + selectedTarget : "Install for " + selectedTarget) +
            "</button>" +
            removeButton +
          "</div>";
        container.appendChild(div);
      }}
      for (const button of container.querySelectorAll('button[data-provider]:not([data-action="remove"])')) {{
        button.addEventListener("click", installModel);
      }}
      for (const button of container.querySelectorAll('button[data-action="remove"]')) {{
        button.addEventListener("click", removeModel);
      }}
    }}

    function effectiveProcessingTarget(item) {{
      const requested = document.getElementById("processing-device").value || "auto";
      if (requested !== "auto") {{
        return requested;
      }}
      return item.recommended_target || "cpu";
    }}

    function isModelSelectableForTarget(item, target) {{
      return item.available && (item.supported_targets || []).includes(target);
    }}

    function renderModelSelect(payload) {{
      const select = document.getElementById("model-select");
      const previous = selectedModelValue(currentSettings);
      select.innerHTML = "";
      const autoOption = document.createElement("option");
      autoOption.value = "auto";
      autoOption.textContent = "Auto recommend";
      select.appendChild(autoOption);
      for (const item of payload.models || []) {{
        const option = document.createElement("option");
        option.value = item.provider + "::" + item.model_id;
        option.textContent = item.label + ((item.installed_targets || []).length ? " [installed]" : "") + (item.recommended ? " [recommended]" : "");
        option.disabled = !isModelSelectableForTarget(item, effectiveProcessingTarget(item));
        select.appendChild(option);
      }}
      const explicitExists = Array.from(select.options).some((option) => option.value === previous);
      if (explicitExists) {{
        select.value = previous;
      }} else if (previous === "auto") {{
        select.value = "auto";
      }} else if (payload.recommended_provider && payload.recommended_model_id) {{
        const recommendedValue = payload.recommended_provider + "::" + payload.recommended_model_id;
        if (Array.from(select.options).some((option) => option.value === recommendedValue)) {{
          select.value = recommendedValue;
        }}
      }}
    }}

    function renderInstallJob(job) {{
      const container = document.getElementById("job-status");
      const stage = document.getElementById("job-stage");
      const message = document.getElementById("job-message");
      const log = document.getElementById("job-log");
      container.hidden = false;
      stage.textContent = "Status: " + job.status + " • Stage: " + job.stage;
      message.textContent = job.message + (job.detail ? " • " + job.detail : "");
      log.textContent = (job.log_lines || []).join("\\n");
    }}

    function renderProgress(job) {{
      if (job.progress_percent === null || job.progress_percent === undefined) {{
        return "";
      }}
      const current = job.progress_current ?? 0;
      const total = job.progress_total ?? 0;
      return (
        '<div class="progress-shell">' +
          '<div class="progress-row">' +
            '<span>Translation progress</span>' +
            '<span>' + job.progress_percent + '% (' + current + '/' + total + ')</span>' +
          '</div>' +
          '<div class="progress-bar"><div class="progress-fill" style="width:' + job.progress_percent + '%"></div></div>' +
        '</div>'
      );
    }}

    function renderSubtitleActivity(payload) {{
      const status = document.getElementById("subtitle-activity-status");
      const container = document.getElementById("subtitle-activity");
      const jobs = payload.jobs || [];
      container.innerHTML = "";
      if (!jobs.length) {{
        status.textContent = "No subtitle generation activity yet.";
        return;
      }}
      status.textContent = "Showing latest " + jobs.length + " subtitle generation request" + (jobs.length === 1 ? "." : "s.");
      for (const job of jobs) {{
        const div = document.createElement("div");
        div.className = "job-status";
        const shortKey = job.cache_key.length > 34 ? job.cache_key.slice(0, 34) + "..." : job.cache_key;
        const sourceName = job.source_subtitle_name || "Unknown source subtitle";
        const mediaName = job.media_filename ? "<div class=\\"meta\\">Media file: " + job.media_filename + "</div>" : "";
        const sourceId = job.source_subtitle_id ? "<div class=\\"meta\\">Source subtitle: " + sourceName + " • id " + job.source_subtitle_id + "</div>" : "<div class=\\"meta\\">Source subtitle: " + sourceName + "</div>";
        div.innerHTML =
          "<div class=\\"job-stage\\">Status: " + job.status + " • Stage: " + job.stage + "</div>" +
          "<div class=\\"meta\\">" + job.message + "</div>" +
          sourceId +
          mediaName +
          renderProgress(job) +
          "<div class=\\"meta\\">" + shortKey + "</div>" +
          "<div class=\\"job-log\\">" + (job.log_lines || []).join("\\n") + "</div>";
        container.appendChild(div);
      }}
    }}

    async function refreshSubtitleActivity() {{
      try {{
        const payload = await loadJson("/api/subtitles/status");
        renderSubtitleActivity(payload);
      }} catch (error) {{
        document.getElementById("subtitle-activity-status").textContent = error.message;
      }}
    }}

    async function pollInstallJob(jobId) {{
      if (activeInstallPoll) {{
        clearInterval(activeInstallPoll);
      }}
      activeInstallJobId = jobId;
      async function tick() {{
        try {{
          const job = await loadJson("/api/models/install/" + jobId);
          renderInstallJob(job);
          document.getElementById("model-status").textContent = job.message;
          if (job.status === "completed") {{
            clearInterval(activeInstallPoll);
            activeInstallPoll = null;
            currentSettings.preferred_provider = job.provider;
            currentSettings.selected_model_id = job.model_id;
            const selection = job.provider + "::" + job.model_id;
            if (Array.from(document.getElementById("model-select").options).some((option) => option.value === selection)) {{
              document.getElementById("model-select").value = selection;
            }}
            renderManifest(currentSettings);
            await evaluateModels();
          }}
          if (job.status === "failed") {{
            clearInterval(activeInstallPoll);
            activeInstallPoll = null;
          }}
        }} catch (error) {{
          document.getElementById("model-status").textContent = error.message;
          clearInterval(activeInstallPoll);
          activeInstallPoll = null;
        }}
      }}
      await tick();
      activeInstallPoll = setInterval(tick, 1200);
    }}

    async function evaluateModels() {{
      const settings = readSettings();
      currentSettings = settings;
      renderManifest(settings);
      document.getElementById("model-status").textContent = "Checking local recommendation and model availability...";
      try {{
        const payload = await loadJson("/api/models?source_lang=" + settings.source_lang + "&target_lang=" + settings.target_lang);
        renderModels(payload);
        currentSettings = readSettings();
        renderManifest(currentSettings);
        document.getElementById("model-status").textContent = "";
      }} catch (error) {{
        document.getElementById("model-status").textContent = error.message;
      }}
    }}

    async function installModel(event) {{
      const provider = event.currentTarget.dataset.provider;
      const modelId = event.currentTarget.dataset.modelId;
      const settings = readSettings();
      document.getElementById("model-status").textContent = "Queueing install for " + provider + "...";
      try {{
        const job = await loadJson("/api/models/install", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            provider: provider,
            model_id: modelId,
            source_lang: settings.source_lang,
            target_lang: settings.target_lang,
            processing_device: settings.processing_device,
            persist_selection: true
          }})
        }});
        await pollInstallJob(job.job_id);
      }} catch (error) {{
        document.getElementById("model-status").textContent = error.message;
      }}
    }}

    async function removeModel(event) {{
      const provider = event.currentTarget.dataset.provider;
      const modelId = event.currentTarget.dataset.modelId;
      const target = event.currentTarget.dataset.target || "auto";
      if (!window.confirm("Remove this installed target from Polylinguist? Shared model caches used by other tools will be left in place.")) {{
        return;
      }}
      document.getElementById("model-status").textContent = "Removing " + provider + " for " + target + "...";
      try {{
        const result = await loadJson("/api/models/remove", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            provider: provider,
            model_id: modelId,
            processing_device: target
          }})
        }});
        const note = (result.notes || []).join(" ");
        document.getElementById("model-status").textContent = result.detail + (note ? " " + note : "");
        await syncSettingsFromServer();
        await refreshSubtitleActivity();
      }} catch (error) {{
        document.getElementById("model-status").textContent = error.message;
      }}
    }}

    async function saveSettings() {{
      const settings = readSettings();
      document.getElementById("settings-status").textContent = "Saving local defaults...";
      try {{
        const payload = await loadJson("/api/settings", {{
          method: "PUT",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(settings)
        }});
        currentSettings = payload.settings;
        renderManifest(currentSettings);
        document.getElementById("settings-status").textContent = "Saved.";
      }} catch (error) {{
        document.getElementById("settings-status").textContent = error.message;
      }}
    }}

    async function syncSettingsFromServer() {{
      const envelope = await loadJson("/api/settings");
      applySettingsToForm(envelope.settings);
      await evaluateModels();
    }}

    function refreshManifestPreview() {{
      currentSettings = readSettings();
      renderManifest(currentSettings);
      if (latestCatalogPayload) {{
        renderModels(latestCatalogPayload);
      }}
    }}

    async function boot() {{
      const [languages, profile, envelope] = await Promise.all([
        loadJson("/api/languages"),
        loadJson("/api/system/profile"),
        loadJson("/api/settings")
      ]);
      fillLanguages(languages);
      renderSystemProfile(profile);
      applySettingsToForm(envelope.settings);
      await evaluateModels();
      await refreshSubtitleActivity();
      setInterval(refreshSubtitleActivity, 1000);
    }}

    document.getElementById("refresh-models").addEventListener("click", evaluateModels);
    document.getElementById("save-settings").addEventListener("click", saveSettings);
    document.getElementById("copy-manifest").addEventListener("click", copyManifestUrl);
    document.getElementById("source-lang").addEventListener("change", refreshManifestPreview);
    document.getElementById("target-lang").addEventListener("change", refreshManifestPreview);
    document.getElementById("format-mode").addEventListener("change", refreshManifestPreview);
    document.getElementById("processing-device").addEventListener("change", refreshManifestPreview);
    document.getElementById("model-select").addEventListener("change", refreshManifestPreview);
    boot().catch((error) => {{
      document.getElementById("model-status").textContent = error.message;
    }});
  </script>
</body>
</html>"""
