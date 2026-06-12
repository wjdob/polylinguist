from pathlib import Path

from fastapi.testclient import TestClient

from polylinguist.config import AppConfig, AppPaths
from polylinguist.schemas import AddonSettings, ModelCatalogResponse, ModelOptionResponse
from polylinguist.services.runtime import AppServices, create_services
from polylinguist.services.system_profile import AcceleratorInfo, SystemProfile
from polylinguist.services.subtitle_sources import SubtitleSourceExtra
from polylinguist.services.subtitles import SubtitleCandidate
from polylinguist.app import create_app


class FakeProvider:
    async def list_candidates(self, media_type, media_id, source_lang, extra, limit=3):
        return [
            SubtitleCandidate(
                subtitle_id="sub1",
                url="https://example.test/sub1.srt",
                lang=source_lang,
                label="Example",
                source="test",
                score=100.0,
                match_label="hash match",
            )
        ]

    async def fetch_subtitle_text(self, candidate):
        return """1
00:00:01,000 --> 00:00:02,000
Hello world
"""


class FakeTranslationManager:
    def __init__(self):
        self.installed = {("marian", "fake-model")}

    def is_installed(self, provider, model_id, device_preference="auto"):
        return (provider, model_id) in self.installed

    def translate_batch(self, request, cues, progress=None):
        if progress:
            progress("translate", "Translating cue 1/1.")
        return [f"translated:{cue}" for cue in cues]

    def install(self, descriptor, progress=None, device_preference="auto"):
        if progress:
            progress("runtime", "Using fake runtime.")
            progress("download", "Pretending to download fake model.")
        self.installed.add((descriptor.provider, descriptor.model_id))
        return "installed"

    def forget_model(self, provider, model_id, target=None):
        self.installed.discard((provider, model_id))

    def clear_runtime_state(self):
        return None


class StaticProfileService:
    def __init__(self, profile: SystemProfile):
        self.profile = profile

    def detect(self) -> SystemProfile:
        return self.profile


def build_test_app(tmp_path: Path):
    paths = AppPaths(
        root=tmp_path / ".polylinguist",
        cache_dir=tmp_path / ".polylinguist" / "cache",
        model_artifacts_dir=tmp_path / ".polylinguist" / "models",
        settings_file=tmp_path / ".polylinguist" / "settings.json",
        installed_models_file=tmp_path / ".polylinguist" / "installed_models.json",
        metadata_cache_file=tmp_path / ".polylinguist" / "cache" / "model_metadata.json",
        generated_subtitles_dir=tmp_path / ".polylinguist" / "cache" / "subtitles",
    )
    services = create_services(paths, AppConfig.detect())
    services.settings_store.save(
        AddonSettings(
            source_lang="eng",
            target_lang="spa",
            preferred_provider="marian",
            selected_model_id="fake-model",
        )
    )
    services.subtitle_provider = FakeProvider()
    services.translation_manager = FakeTranslationManager()
    services.system_profile_service = StaticProfileService(
        SystemProfile("windows", "amd64", 8, 16.0, 50.0, False, False)
    )
    services.model_catalog.list_models = lambda source_lang, target_lang, profile, refresh=False: ModelCatalogResponse(
        source_lang=source_lang,
        target_lang=target_lang,
        recommended_provider="marian",
        recommended_model_id="fake-model",
        profile=SystemProfile("windows", "amd64", 8, 16.0, 50.0, False, False).to_response(),
        models=[
            ModelOptionResponse(
                provider="marian",
                model_id="fake-model",
                label="Fake Marian",
                source_lang=source_lang,
                target_lang=target_lang,
                size_mb=1,
                available=True,
                direct=True,
                installed=True,
                installed_targets=["cpu"],
                supported_targets=["cpu"],
                recommended_target="cpu",
                availability_reason="Fake Marian supports CPU.",
                license="test",
                recommended=True,
            )
        ],
    )
    return create_app(services)


def build_uninstalled_test_app(tmp_path: Path):
    paths = AppPaths(
        root=tmp_path / ".polylinguist",
        cache_dir=tmp_path / ".polylinguist" / "cache",
        model_artifacts_dir=tmp_path / ".polylinguist" / "models",
        settings_file=tmp_path / ".polylinguist" / "settings.json",
        installed_models_file=tmp_path / ".polylinguist" / "installed_models.json",
        metadata_cache_file=tmp_path / ".polylinguist" / "cache" / "model_metadata.json",
        generated_subtitles_dir=tmp_path / ".polylinguist" / "cache" / "subtitles",
    )
    services = create_services(paths, AppConfig.detect())
    services.settings_store.save(
        AddonSettings(
            source_lang="eng",
            target_lang="pol",
            preferred_provider="marian",
            selected_model_id="Helsinki-NLP/opus-mt-en-ine",
            processing_device="cpu",
        )
    )
    services.subtitle_provider = FakeProvider()
    services.translation_manager = FakeTranslationManager()
    services.translation_manager.installed = set()
    services.system_profile_service = StaticProfileService(
        SystemProfile("windows", "amd64", 8, 16.0, 50.0, True, False)
    )
    services.model_catalog.list_models = lambda source_lang, target_lang, profile, refresh=False: ModelCatalogResponse(
        source_lang=source_lang,
        target_lang=target_lang,
        recommended_provider="marian",
        recommended_model_id="Helsinki-NLP/opus-mt-en-ine",
        profile=SystemProfile("windows", "amd64", 8, 16.0, 50.0, True, False).to_response(),
        models=[
            ModelOptionResponse(
                provider="marian",
                model_id="Helsinki-NLP/opus-mt-en-ine",
                label="Fake Marian Fallback",
                source_lang=source_lang,
                target_lang=target_lang,
                size_mb=320,
                available=True,
                direct=False,
                installed=False,
                installed_targets=[],
                supported_targets=["cpu"],
                recommended_target="cpu",
                availability_reason="Fake Marian supports CPU.",
                license="test",
                recommended=True,
            )
        ],
    )
    return create_app(services)


def build_unsupported_target_app(tmp_path: Path):
    paths = AppPaths(
        root=tmp_path / ".polylinguist",
        cache_dir=tmp_path / ".polylinguist" / "cache",
        model_artifacts_dir=tmp_path / ".polylinguist" / "models",
        settings_file=tmp_path / ".polylinguist" / "settings.json",
        installed_models_file=tmp_path / ".polylinguist" / "installed_models.json",
        metadata_cache_file=tmp_path / ".polylinguist" / "cache" / "model_metadata.json",
        generated_subtitles_dir=tmp_path / ".polylinguist" / "cache" / "subtitles",
    )
    services = create_services(paths, AppConfig.detect())
    services.settings_store.save(
        AddonSettings(
            source_lang="eng",
            target_lang="pol",
            preferred_provider="marian",
            selected_model_id="Helsinki-NLP/opus-mt-en-ine",
            processing_device="cuda",
        )
    )
    services.subtitle_provider = FakeProvider()
    services.translation_manager = FakeTranslationManager()
    services.translation_manager.installed = set()
    services.system_profile_service = StaticProfileService(
        SystemProfile("windows", "amd64", 8, 16.0, 50.0, False, False)
    )
    services.model_catalog.list_models = lambda source_lang, target_lang, profile, refresh=False: ModelCatalogResponse(
        source_lang=source_lang,
        target_lang=target_lang,
        recommended_provider="marian",
        recommended_model_id="Helsinki-NLP/opus-mt-en-ine",
        profile=profile.to_response(),
        models=[
            ModelOptionResponse(
                provider="marian",
                model_id="Helsinki-NLP/opus-mt-en-ine",
                label="Fake Marian Fallback",
                source_lang=source_lang,
                target_lang=target_lang,
                size_mb=320,
                available=True,
                direct=False,
                installed=False,
                installed_targets=[],
                supported_targets=["cpu", "cuda"],
                recommended_target="cpu",
                availability_reason="Fake Marian supports CPU and CUDA.",
                license="test",
                recommended=True,
            )
        ],
    )
    return create_app(services)


def build_remote_admin_app(tmp_path: Path):
    paths = AppPaths(
        root=tmp_path / ".polylinguist",
        cache_dir=tmp_path / ".polylinguist" / "cache",
        model_artifacts_dir=tmp_path / ".polylinguist" / "models",
        settings_file=tmp_path / ".polylinguist" / "settings.json",
        installed_models_file=tmp_path / ".polylinguist" / "installed_models.json",
        metadata_cache_file=tmp_path / ".polylinguist" / "cache" / "model_metadata.json",
        generated_subtitles_dir=tmp_path / ".polylinguist" / "cache" / "subtitles",
    )
    config = AppConfig(
        bind_host="0.0.0.0",
        bind_port=8000,
        public_base_url="https://subs.example.test",
        admin_token="secret-token",
    )
    services = create_services(paths, config)
    services.settings_store.save(
        AddonSettings(
            source_lang="eng",
            target_lang="spa",
            preferred_provider="marian",
            selected_model_id="fake-model",
        )
    )
    services.subtitle_provider = FakeProvider()
    services.translation_manager = FakeTranslationManager()
    services.system_profile_service = StaticProfileService(
        SystemProfile(
            "windows",
            "amd64",
            8,
            16.0,
            50.0,
            False,
            False,
            accelerators=(AcceleratorInfo(vendor="intel", name="Arc A750", supported_targets=("openvino_gpu",)),),
        )
    )
    services.model_catalog.list_models = lambda source_lang, target_lang, profile, refresh=False: ModelCatalogResponse(
        source_lang=source_lang,
        target_lang=target_lang,
        recommended_provider="marian",
        recommended_model_id="fake-model",
        profile=SystemProfile(
            "windows",
            "amd64",
            8,
            16.0,
            50.0,
            False,
            False,
            accelerators=(AcceleratorInfo(vendor="intel", name="Arc A750", supported_targets=("openvino_gpu",)),),
        ).to_response(),
        models=[
            ModelOptionResponse(
                provider="marian",
                model_id="fake-model",
                label="Fake Marian",
                source_lang=source_lang,
                target_lang=target_lang,
                size_mb=1,
                available=True,
                direct=True,
                installed=True,
                installed_targets=["openvino_gpu"],
                supported_targets=["cpu", "openvino_gpu"],
                recommended_target="openvino_gpu",
                availability_reason="Fake Marian supports OpenVINO GPU.",
                license="test",
                recommended=True,
            )
        ],
    )
    return create_app(services)


def test_manifest_and_subtitle_flow(tmp_path: Path):
    app = build_test_app(tmp_path)
    client = TestClient(app)

    manifest = client.get("/manifest.json")
    assert manifest.status_code == 200
    links = manifest.json()["links"]
    assert "manifest" in links

    config_token = links["manifest"].split("/")[-2]
    subtitles = client.get(f"/{config_token}/subtitles/movie/tt1234567.json")
    assert subtitles.status_code == 200
    payload = subtitles.json()
    assert len(payload["subtitles"]) == 1
    assert payload["subtitles"][0]["lang"] == "eng+spa"
    assert payload["cacheMaxAge"] == 60
    subtitle_url = payload["subtitles"][0]["url"]
    generated = client.get(subtitle_url.replace("http://testserver", ""))
    assert generated.status_code == 200
    assert generated.headers["cache-control"] == "no-store"
    assert "translated:Hello world" in generated.text
    assert "<i>" not in generated.text
    assert "00:00:01,000 --> 00:00:02,000" in generated.text
    assert "Hello world" in generated.text


def test_configure_page_exposes_explicit_online_refresh(tmp_path: Path):
    app = build_test_app(tmp_path)
    client = TestClient(app)

    response = client.get("/configure")

    assert response.status_code == 200
    assert "Refresh availability" in response.text
    assert 'document.getElementById("refresh-models-online")' in response.text


def test_subtitle_flow_accepts_stremio_extra_path(tmp_path: Path):
    app = build_test_app(tmp_path)
    client = TestClient(app)

    manifest = client.get("/manifest.json")
    config_token = manifest.json()["links"]["manifest"].split("/")[-2]
    subtitles = client.get(
        f"/{config_token}/subtitles/movie/tt1234567/videoHash=abc123&videoSize=42&filename=Example%20Movie.mkv.json"
    )

    assert subtitles.status_code == 200
    assert len(subtitles.json()["subtitles"]) == 1


def test_install_job_status(tmp_path: Path):
    app = build_test_app(tmp_path)
    client = TestClient(app)
    response = client.post(
        "/api/models/install",
        json={
            "provider": "marian",
            "model_id": "fake-model",
            "source_lang": "eng",
            "target_lang": "spa",
            "processing_device": "cpu",
            "persist_selection": True,
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    status = client.get(f"/api/models/install/{job_id}")
    assert status.status_code == 200
    payload = status.json()
    assert payload["status"] in {"running", "completed"}
    assert payload["provider"] == "marian"


def test_remove_model_clears_selected_model(tmp_path: Path):
    app = build_test_app(tmp_path)
    client = TestClient(app)

    response = client.post(
        "/api/models/remove",
        json={
            "provider": "marian",
            "model_id": "fake-model",
            "processing_device": "cpu",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Removed" in payload["detail"]
    assert payload["notes"]

    settings = client.get("/api/settings").json()["settings"]
    assert settings["preferred_provider"] == "auto"
    assert settings["selected_model_id"] is None


def test_uninstall_local_data_resets_settings(tmp_path: Path):
    paths = AppPaths(
        root=tmp_path / ".polylinguist",
        cache_dir=tmp_path / ".polylinguist" / "cache",
        model_artifacts_dir=tmp_path / ".polylinguist" / "models",
        settings_file=tmp_path / ".polylinguist" / "settings.json",
        installed_models_file=tmp_path / ".polylinguist" / "installed_models.json",
        metadata_cache_file=tmp_path / ".polylinguist" / "cache" / "model_metadata.json",
        generated_subtitles_dir=tmp_path / ".polylinguist" / "cache" / "subtitles",
    )
    services = create_services(paths, AppConfig.detect())
    services.settings_store.save(
        AddonSettings(
            source_lang="eng",
            target_lang="spa",
            preferred_provider="marian",
            selected_model_id="fake-model",
        )
    )
    paths.generated_subtitles_dir.mkdir(parents=True, exist_ok=True)
    (paths.generated_subtitles_dir / "cached.srt").write_text("1", encoding="utf-8")

    result = services.uninstall_local_data()

    assert "first-run defaults" in result["detail"]
    assert result["notes"]
    settings = services.settings_store.load().settings
    assert settings.preferred_provider == "auto"
    assert settings.selected_model_id is None
    assert settings.source_lang == "eng"


def test_settings_round_trip_includes_processing_device(tmp_path: Path):
    app = build_test_app(tmp_path)
    client = TestClient(app)

    response = client.put(
        "/api/settings",
        json={
            "source_lang": "eng",
            "target_lang": "spa",
            "preferred_provider": "marian",
            "selected_model_id": "fake-model",
            "processing_device": "cpu",
            "format_mode": "dual",
        },
    )

    assert response.status_code == 200
    assert response.json()["settings"]["processing_device"] == "cpu"


def test_runtime_diagnostics_reports_runtimes(tmp_path: Path):
    app = build_test_app(tmp_path)
    client = TestClient(app)

    response = client.get("/api/diagnostics/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["os"] == "windows"
    assert any(item["provider"] == "marian" and item["target"] == "cpu" for item in payload["runtimes"])


def test_runtime_diagnostics_prefers_compatible_openvino_worker(tmp_path: Path, monkeypatch):
    app = build_remote_admin_app(tmp_path)
    client = TestClient(app)

    fake_runtime = type(
        "Runtime",
        (),
        {
            "executable": r"C:\\Python313\\python.exe",
            "current": False,
            "has_cuda": False,
            "has_directml": False,
            "has_openvino_gpu": False,
            "python_version": "3.13.9",
        },
    )()

    monkeypatch.setattr("polylinguist.services.runtime.compatible_runtime_block_reason", lambda target, system_name=None: None)
    monkeypatch.setattr(
        "polylinguist.services.runtime.resolve_runtime_for_target",
        lambda target, requirements, prefer_cuda=False, system_name=None: fake_runtime,
    )
    monkeypatch.setattr(
        "polylinguist.services.runtime.runtime_metadata_snapshot",
        lambda runtime, requirements: {
            "python_executable": runtime.executable,
            "python_version": runtime.python_version,
            "has_cuda": runtime.has_cuda,
            "has_directml": runtime.has_directml,
            "has_openvino_gpu": runtime.has_openvino_gpu,
            "package_versions": {},
        },
    )

    response = client.get(
        "/api/diagnostics/runtime",
        headers={"X-Polylinguist-Admin-Token": "secret-token"},
    )

    assert response.status_code == 200
    payload = response.json()
    openvino_row = next(item for item in payload["runtimes"] if item["provider"] == "marian" and item["target"] == "openvino_gpu")
    assert openvino_row["selected_python"] == r"C:\\Python313\\python.exe"


def test_subtitle_generation_activity(tmp_path: Path):
    app = build_test_app(tmp_path)
    client = TestClient(app)

    manifest = client.get("/manifest.json")
    config_token = manifest.json()["links"]["manifest"].split("/")[-2]
    subtitles = client.get(f"/{config_token}/subtitles/movie/tt1234567.json").json()
    subtitle_url = subtitles["subtitles"][0]["url"]
    generated = client.get(subtitle_url.replace("http://testserver", ""))
    assert generated.status_code == 200

    activity = client.get("/api/subtitles/status")
    assert activity.status_code == 200
    jobs = activity.json()["jobs"]
    assert jobs
    assert jobs[0]["status"] == "completed"
    assert jobs[0]["source_subtitle_name"] == "Example"
    assert jobs[0]["progress_percent"] == 100
    assert any("Translating cue 1/1." in line for line in jobs[0]["log_lines"])


def test_uninstalled_model_returns_setup_placeholder(tmp_path: Path):
    app = build_uninstalled_test_app(tmp_path)
    client = TestClient(app)

    manifest = client.get("/manifest.json")
    config_token = manifest.json()["links"]["manifest"].split("/")[-2]
    subtitles = client.get(f"/{config_token}/subtitles/movie/tt1234567.json")

    assert subtitles.status_code == 200
    payload = subtitles.json()
    assert len(payload["subtitles"]) == 1
    assert "setup required" in payload["subtitles"][0]["label"].lower()

    generated = client.get(payload["subtitles"][0]["url"].replace("http://testserver", ""))
    assert generated.status_code == 200
    assert "Polylinguist setup required." in generated.text
    assert "00:00:00,000 --> 99:59:59,000" in generated.text


def test_unsupported_target_returns_configuration_placeholder(tmp_path: Path):
    app = build_unsupported_target_app(tmp_path)
    client = TestClient(app)

    manifest = client.get("/manifest.json")
    config_token = manifest.json()["links"]["manifest"].split("/")[-2]
    subtitles = client.get(f"/{config_token}/subtitles/movie/tt1234567.json")

    assert subtitles.status_code == 200
    payload = subtitles.json()
    assert len(payload["subtitles"]) == 1
    assert "configuration issue" in payload["subtitles"][0]["label"].lower()

    generated = client.get(payload["subtitles"][0]["url"].replace("http://testserver", ""))
    assert generated.status_code == 200
    assert "Polylinguist configuration needs attention." in generated.text


def test_manifest_has_cors_headers(tmp_path: Path):
    app = build_test_app(tmp_path)
    client = TestClient(app)
    response = client.get(
        "/manifest.json",
        headers={"Origin": "https://app.strem.io"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_remote_manifest_uses_public_base_url(tmp_path: Path):
    app = build_remote_admin_app(tmp_path)
    client = TestClient(app)

    manifest = client.get("/manifest.json")
    assert manifest.status_code == 200
    payload = manifest.json()
    assert payload["links"]["manifest"].startswith("https://subs.example.test/")
    assert payload["links"]["configure"] == "https://subs.example.test/configure"


def test_admin_token_protects_api_routes(tmp_path: Path):
    app = build_remote_admin_app(tmp_path)
    client = TestClient(app)

    unauthorized = client.get("/api/settings")
    assert unauthorized.status_code == 401

    authorized = client.get(
        "/api/settings",
        headers={"X-Polylinguist-Admin-Token": "secret-token"},
    )
    assert authorized.status_code == 200
