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

from polylinguist.config import AppPaths
from polylinguist.schemas import (
    AddonSettings,
    InstallJobResponse,
    LanguageResponse,
    ModelCatalogResponse,
    ModelInstallRequest,
    ModelInstallResponse,
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
    app.state.services = services or create_services(AppPaths.detect())

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> str:
        return render_configure_page(str(request.base_url).rstrip("/"))

    @app.get("/configure", response_class=HTMLResponse)
    async def configure(request: Request) -> str:
        return render_configure_page(str(request.base_url).rstrip("/"))

    @app.get("/api/languages", response_model=list[LanguageResponse])
    async def get_languages() -> list[LanguageResponse]:
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
        services = get_services(request)
        return services.system_profile_service.detect().to_response().model_dump()

    @app.get("/api/settings", response_model=SettingsEnvelope)
    async def get_settings(request: Request) -> SettingsEnvelope:
        return get_services(request).settings_store.load()

    @app.put("/api/settings", response_model=SettingsEnvelope)
    async def put_settings(request: Request, settings: AddonSettings) -> SettingsEnvelope:
        return get_services(request).settings_store.save(settings)

    @app.get("/api/models", response_model=ModelCatalogResponse)
    async def get_models(
        request: Request,
        source_lang: str = Query(...),
        target_lang: str = Query(...),
    ) -> ModelCatalogResponse:
        services = get_services(request)
        return services.model_catalog.list_models(
            source_lang,
            target_lang,
            services.system_profile_service.detect(),
        )

    @app.post("/api/models/install", response_model=InstallJobResponse)
    async def install_model(request: Request, payload: ModelInstallRequest) -> InstallJobResponse:
        services = get_services(request)
        catalog = services.model_catalog.list_models(
            payload.source_lang,
            payload.target_lang,
            services.system_profile_service.detect(),
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
                payload.processing_device,
            ),
        )
        return InstallJobResponse.model_validate(job.to_dict())

    @app.get("/api/models/install/{job_id}", response_model=InstallJobResponse)
    async def get_install_job(request: Request, job_id: str) -> InstallJobResponse:
        services = get_services(request)
        job = services.install_job_manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Install job not found.")
        return InstallJobResponse.model_validate(job.to_dict())

    @app.get("/api/subtitles/status")
    async def get_subtitle_activity(request: Request) -> dict[str, Any]:
        services = get_services(request)
        return {
            "jobs": [job.to_dict() for job in services.subtitle_generation_tracker.recent()],
        }

    @app.get("/api/subtitles/status/{cache_key}")
    async def get_subtitle_status(request: Request, cache_key: str) -> dict[str, Any]:
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
        base_url = str(request.base_url).rstrip("/")
        config_token = encode_config(settings)
        return build_manifest(base_url, settings, config_token)

    @app.get("/{config_token}/manifest.json")
    async def configured_manifest(request: Request, config_token: str) -> dict[str, Any]:
        settings = decode_config(config_token)
        base_url = str(request.base_url).rstrip("/")
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
                    "Open the Polylinguist configure page for progress, then re-select this subtitle when it is ready.",
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
    items = await services.build_subtitle_options(
        settings=settings,
        media_type=media_type,
        media_id=media_id,
        extra=SubtitleSourceExtra(
            filename=filename or extra_params.get("filename"),
            video_size=videoSize or extra_params.get("videoSize"),
            video_hash=videoHash or extra_params.get("videoHash"),
        ),
        base_url=str(request.base_url).rstrip("/"),
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
    if device == "cuda":
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


def render_configure_page(base_url: str) -> str:
    escaped_url = quote(base_url, safe=":/")
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
    button.secondary {{
      background: transparent;
      color: var(--ink);
      border: 1px solid var(--line-strong);
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
    const apiBase = "{escaped_url}";
    let currentSettings = null;
    let activeInstallJobId = null;
    let activeInstallPoll = null;
    let availableModels = [];
    let systemProfile = null;

    async function loadJson(path, options) {{
      const response = await fetch(apiBase + path, options);
      const contentType = response.headers.get("content-type") || "";
      const payload = contentType.includes("application/json") ? await response.json() : await response.text();
      if (!response.ok) {{
        throw new Error(payload.detail || payload || "Request failed");
      }}
      return payload;
    }}

    function manifestUrl(settings) {{
      const encoded = btoa(JSON.stringify(settings)).replace(/\\+/g, "-").replace(/\\//g, "_").replace(/=+$/g, "");
      return apiBase + "/" + encoded + "/manifest.json";
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
      document.getElementById("system-profile").textContent =
        profile.os + " / " + profile.arch +
        " • " + profile.cpu_cores + " cores" +
        " • " + profile.total_ram_gb.toFixed(1) + " GB RAM" +
        " • " + profile.free_disk_gb.toFixed(1) + " GB free" +
        " • tier: " + profile.tier +
        " • CUDA: " + (profile.has_cuda ? "yes" : "no") +
        " • MPS: " + (profile.has_mps ? "yes" : "no");
      const gpuOption = document.querySelector('#processing-device option[value="cuda"]');
      gpuOption.disabled = !profile.has_cuda;
      if (!profile.has_cuda && document.getElementById("processing-device").value === "cuda") {{
        document.getElementById("processing-device").value = "auto";
      }}
    }}

    function renderModels(payload) {{
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
        const badge = item.recommended ? "Recommended for this machine" : item.available ? "Available" : "Unavailable";
        div.innerHTML =
          "<strong>" + item.label + "</strong>" +
          "<div class=\\"meta\\">" + badge + " • " + item.size_mb + " MB • " + item.license + "</div>" +
          note +
          "<div class=\\"install-row\\">" +
            "<button type=\\"button\\" " + (item.available ? "" : "disabled") + " data-provider=\\"" + item.provider + "\\" data-model-id=\\"" + item.model_id + "\\">" +
              (item.installed ? "Reinstall" : "Install") +
            "</button>" +
          "</div>";
        container.appendChild(div);
      }}
      for (const button of container.querySelectorAll("button[data-provider]")) {{
        button.addEventListener("click", installModel);
      }}
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
        option.textContent = item.label + (item.installed ? " [installed]" : "") + (item.recommended ? " [recommended]" : "");
        option.disabled = !item.available;
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

    function refreshManifestPreview() {{
      currentSettings = readSettings();
      renderManifest(currentSettings);
    }}

    async function boot() {{
      const [languages, profile, envelope] = await Promise.all([
        loadJson("/api/languages"),
        loadJson("/api/system/profile"),
        loadJson("/api/settings")
      ]);
      fillLanguages(languages);
      renderSystemProfile(profile);
      currentSettings = envelope.settings;
      document.getElementById("source-lang").value = currentSettings.source_lang;
      document.getElementById("target-lang").value = currentSettings.target_lang;
      document.getElementById("format-mode").value = currentSettings.format_mode;
      document.getElementById("processing-device").value =
        currentSettings.processing_device === "cuda" && !profile.has_cuda
          ? "auto"
          : (currentSettings.processing_device || "auto");
      renderManifest(currentSettings);
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
