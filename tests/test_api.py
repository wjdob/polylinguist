from pathlib import Path

from fastapi.testclient import TestClient

from polylinguist.config import AppPaths
from polylinguist.schemas import AddonSettings, ModelCatalogResponse, ModelOptionResponse
from polylinguist.services.runtime import AppServices, create_services
from polylinguist.services.system_profile import SystemProfile
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

    def is_installed(self, provider, model_id):
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


def build_test_app(tmp_path: Path):
    paths = AppPaths(
        root=tmp_path / ".polylinguist",
        cache_dir=tmp_path / ".polylinguist" / "cache",
        settings_file=tmp_path / ".polylinguist" / "settings.json",
        installed_models_file=tmp_path / ".polylinguist" / "installed_models.json",
        metadata_cache_file=tmp_path / ".polylinguist" / "cache" / "model_metadata.json",
        generated_subtitles_dir=tmp_path / ".polylinguist" / "cache" / "subtitles",
    )
    services = create_services(paths)
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
    services.model_catalog.list_models = lambda source_lang, target_lang, profile: ModelCatalogResponse(
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
        settings_file=tmp_path / ".polylinguist" / "settings.json",
        installed_models_file=tmp_path / ".polylinguist" / "installed_models.json",
        metadata_cache_file=tmp_path / ".polylinguist" / "cache" / "model_metadata.json",
        generated_subtitles_dir=tmp_path / ".polylinguist" / "cache" / "subtitles",
    )
    services = create_services(paths)
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
    services.model_catalog.list_models = lambda source_lang, target_lang, profile: ModelCatalogResponse(
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


def test_manifest_has_cors_headers(tmp_path: Path):
    app = build_test_app(tmp_path)
    client = TestClient(app)
    response = client.get(
        "/manifest.json",
        headers={"Origin": "https://app.strem.io"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
