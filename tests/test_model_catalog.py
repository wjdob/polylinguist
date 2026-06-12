from pathlib import Path

from polylinguist.services.model_catalog import ModelCatalogService
from polylinguist.services.model_registry import InstalledModelRegistry
from polylinguist.services.system_profile import AcceleratorInfo, SystemProfile


def make_catalog(tmp_path: Path) -> ModelCatalogService:
    registry = InstalledModelRegistry(tmp_path / "installed.json")
    return ModelCatalogService(tmp_path / "meta.json", registry, tmp_path / "models")


def test_argos_direct_pair(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {("en", "es"): 80})
    monkeypatch.setattr(catalog, "_probe_huggingface_model", lambda model_id, refresh=False: (False, None, "cached"))
    profile = SystemProfile("windows", "amd64", 8, 16.0, 50.0, False, False)
    response = catalog.list_models("eng", "spa", profile)
    argos = next(model for model in response.models if model.provider == "argos")
    assert argos.available is True
    assert argos.direct is True


def test_argos_pivot_pair(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {("fr", "en"): 60, ("en", "tr"): 100})
    monkeypatch.setattr(catalog, "_probe_huggingface_model", lambda model_id, refresh=False: (False, None, "cached"))
    profile = SystemProfile("windows", "amd64", 4, 8.0, 20.0, False, False)
    response = catalog.list_models("fre", "tur", profile)
    argos = next(model for model in response.models if model.provider == "argos")
    assert argos.available is True
    assert argos.direct is False
    assert argos.install_strategy == "pivot"


def test_marian_and_nllb_availability(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {})
    monkeypatch.setattr(catalog, "_probe_huggingface_model", lambda model_id, refresh=False: (True, None, "cached"))
    profile = SystemProfile("windows", "amd64", 8, 16.0, 20.0, False, False)
    response = catalog.list_models("eng", "spa", profile)
    marian = next(model for model in response.models if model.provider == "marian")
    nllb = next(model for model in response.models if model.provider == "nllb")
    assert marian.available is True
    assert nllb.available is True
    assert "cpu" in marian.supported_targets
    assert "cuda" not in marian.supported_targets


def test_marian_english_polish_falls_back_to_en_ine(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {})
    monkeypatch.setattr(
        catalog,
        "_probe_huggingface_model",
        lambda model_id, refresh=False: (model_id == "Helsinki-NLP/opus-mt-en-ine", None, "cached"),
    )
    monkeypatch.setattr("polylinguist.services.model_catalog.compatible_runtime_block_reason", lambda target, system_name=None: None)
    profile = SystemProfile("windows", "amd64", 8, 16.0, 20.0, True, False)
    response = catalog.list_models("eng", "pol", profile)
    marian = next(model for model in response.models if model.provider == "marian")
    assert marian.available is True
    assert marian.model_id == "Helsinki-NLP/opus-mt-en-ine"
    assert marian.direct is False
    assert "cuda" in marian.supported_targets


def test_marian_english_turkish_falls_back_to_en_trk(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {})
    monkeypatch.setattr(
        catalog,
        "_probe_huggingface_model",
        lambda model_id, refresh=False: (model_id == "Helsinki-NLP/opus-mt-en-trk", None, "cached"),
    )
    profile = SystemProfile("windows", "amd64", 8, 16.0, 20.0, True, False)
    response = catalog.list_models("eng", "tur", profile)
    marian = next(model for model in response.models if model.provider == "marian")
    assert marian.available is True
    assert marian.model_id == "Helsinki-NLP/opus-mt-en-trk"
    assert marian.direct is False


def test_low_machine_prefers_argos(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {("en", "es"): 80})
    monkeypatch.setattr(catalog, "_probe_huggingface_model", lambda model_id, refresh=False: (True, None, "cached"))
    profile = SystemProfile("windows", "amd64", 2, 4.0, 10.0, False, False)
    response = catalog.list_models("eng", "spa", profile)
    assert response.recommended_provider == "argos"


def test_marian_exposes_directml_and_openvino_targets(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {})
    monkeypatch.setattr(catalog, "_probe_huggingface_model", lambda model_id, refresh=False: (True, None, "cached"))
    monkeypatch.setattr("polylinguist.services.model_catalog.compatible_runtime_block_reason", lambda target, system_name=None: None)
    profile = SystemProfile(
        "windows",
        "amd64",
        8,
        16.0,
        20.0,
        False,
        False,
        accelerators=(
            AcceleratorInfo(vendor="amd", name="Radeon RX", supported_targets=("directml",)),
            AcceleratorInfo(vendor="intel", name="Arc A750", supported_targets=("openvino_gpu",)),
        ),
    )
    response = catalog.list_models("eng", "spa", profile)
    marian = next(model for model in response.models if model.provider == "marian")
    assert "directml" in marian.supported_targets
    assert "openvino_gpu" in marian.supported_targets
    assert marian.recommended_target == "openvino_gpu"


def test_marian_hides_openvino_without_compatible_python_runtime(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {})
    monkeypatch.setattr(catalog, "_probe_huggingface_model", lambda model_id, refresh=False: (True, None, "cached"))
    monkeypatch.setattr(
        "polylinguist.services.model_catalog.compatible_runtime_block_reason",
        lambda target, system_name=None: (
            "OpenVINO GPU on Windows requires a discovered Python 3.13 or earlier runtime. Polylinguist did not find one."
            if target == "openvino_gpu"
            else None
        ),
    )
    profile = SystemProfile(
        "windows",
        "amd64",
        8,
        16.0,
        20.0,
        False,
        False,
        accelerators=(AcceleratorInfo(vendor="intel", name="Arc A750", supported_targets=("openvino_gpu",)),),
    )

    response = catalog.list_models("eng", "spa", profile)

    marian = next(model for model in response.models if model.provider == "marian")
    assert "openvino_gpu" not in marian.supported_targets


def test_removed_marian_cpu_target_hides_shared_hf_cache(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    catalog.registry.mark_removed("marian", "Helsinki-NLP/opus-mt-en-ine")
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {})
    monkeypatch.setattr(
        catalog,
        "_probe_huggingface_model",
        lambda model_id, refresh=False: (model_id == "Helsinki-NLP/opus-mt-en-ine", None, "cached"),
    )
    monkeypatch.setattr("polylinguist.services.model_catalog.hf_model_cache_exists", lambda model_id: True)
    profile = SystemProfile("windows", "amd64", 8, 16.0, 20.0, True, False)

    response = catalog.list_models("eng", "pol", profile)

    marian = next(model for model in response.models if model.provider == "marian")
    assert marian.installed is False
    assert "cpu" not in marian.installed_targets
    assert "cuda" not in marian.installed_targets


def test_list_models_is_offline_safe_without_refresh(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    called = []

    def fail_if_called(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("network should not be used for offline-safe model reads")

    monkeypatch.setattr("polylinguist.services.model_catalog.httpx.get", fail_if_called)
    profile = SystemProfile("windows", "amd64", 8, 16.0, 20.0, False, False)

    response = catalog.list_models("eng", "spa", profile)

    assert response.models
    assert called == []
    marian = next(model for model in response.models if model.provider == "marian")
    assert marian.available is True
    assert "offline heuristic" in (marian.note or "").lower()


def test_refresh_huggingface_probe_persists_timestamp(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)

    class Response:
        status_code = 200

    monkeypatch.setattr("polylinguist.services.model_catalog.httpx.get", lambda *args, **kwargs: Response())

    exists, checked_at, source = catalog._probe_huggingface_model("Helsinki-NLP/opus-mt-en-ine", refresh=True)

    assert exists is True
    assert checked_at is not None
    assert source == "refreshed"
