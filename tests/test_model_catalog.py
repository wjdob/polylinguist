from pathlib import Path

from polylinguist.services.model_catalog import ModelCatalogService
from polylinguist.services.model_registry import InstalledModelRegistry
from polylinguist.services.system_profile import SystemProfile


def make_catalog(tmp_path: Path) -> ModelCatalogService:
    registry = InstalledModelRegistry(tmp_path / "installed.json")
    return ModelCatalogService(tmp_path / "meta.json", registry)


def test_argos_direct_pair(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {("en", "es"): 80})
    monkeypatch.setattr(catalog, "_probe_huggingface_model", lambda model_id: False)
    profile = SystemProfile("windows", "amd64", 8, 16.0, 50.0, False, False)
    response = catalog.list_models("eng", "spa", profile)
    argos = next(model for model in response.models if model.provider == "argos")
    assert argos.available is True
    assert argos.direct is True


def test_argos_pivot_pair(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {("fr", "en"): 60, ("en", "tr"): 100})
    monkeypatch.setattr(catalog, "_probe_huggingface_model", lambda model_id: False)
    profile = SystemProfile("windows", "amd64", 4, 8.0, 20.0, False, False)
    response = catalog.list_models("fre", "tur", profile)
    argos = next(model for model in response.models if model.provider == "argos")
    assert argos.available is True
    assert argos.direct is False
    assert argos.install_strategy == "pivot"


def test_marian_and_nllb_availability(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {})
    monkeypatch.setattr(catalog, "_probe_huggingface_model", lambda model_id: True)
    profile = SystemProfile("windows", "amd64", 8, 16.0, 20.0, False, False)
    response = catalog.list_models("eng", "spa", profile)
    marian = next(model for model in response.models if model.provider == "marian")
    nllb = next(model for model in response.models if model.provider == "nllb")
    assert marian.available is True
    assert nllb.available is True


def test_marian_english_polish_falls_back_to_en_ine(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {})
    monkeypatch.setattr(
        catalog,
        "_probe_huggingface_model",
        lambda model_id: model_id == "Helsinki-NLP/opus-mt-en-ine",
    )
    profile = SystemProfile("windows", "amd64", 8, 16.0, 20.0, True, False)
    response = catalog.list_models("eng", "pol", profile)
    marian = next(model for model in response.models if model.provider == "marian")
    assert marian.available is True
    assert marian.model_id == "Helsinki-NLP/opus-mt-en-ine"
    assert marian.direct is False


def test_marian_english_turkish_falls_back_to_en_trk(monkeypatch, tmp_path: Path):
    catalog = make_catalog(tmp_path)
    monkeypatch.setattr(catalog, "_fetch_argos_index", lambda: {})
    monkeypatch.setattr(
        catalog,
        "_probe_huggingface_model",
        lambda model_id: model_id == "Helsinki-NLP/opus-mt-en-trk",
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
    monkeypatch.setattr(catalog, "_probe_huggingface_model", lambda model_id: True)
    profile = SystemProfile("windows", "amd64", 2, 4.0, 10.0, False, False)
    response = catalog.list_models("eng", "spa", profile)
    assert response.recommended_provider == "argos"
