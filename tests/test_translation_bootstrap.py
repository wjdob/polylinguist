import os
import sys
from pathlib import Path

from polylinguist.services.model_catalog import ModelDescriptor
from polylinguist.services.model_registry import InstalledModelRegistry
from polylinguist.services.translation import (
    MarianTranslator,
    _discover_python_executables,
    _ensure_python_packages,
    _ensure_openvino_python_compatibility,
    _configure_huggingface_windows_cache,
    _hf_runtime_packages,
    _load_openvino_seq2seq_class,
    _package_name,
    _package_runtime_ready,
    _module_available,
    _module_name_for_package,
    _openvino_runtime_packages,
    _ort_runtime_packages,
    _resolve_device_preference,
    _resolve_runtime,
    _wrap_huggingface_windows_privilege_error,
    TranslationError,
)


def test_module_name_mapping():
    assert _module_name_for_package("huggingface-hub") == "huggingface_hub"
    assert _module_name_for_package("pillow") == "PIL"
    assert _module_name_for_package("sentencepiece") == "sentencepiece"
    assert _module_name_for_package("onnxruntime-directml") == "onnxruntime"
    assert _module_name_for_package("optimum-intel") == "optimum.intel"
    assert _module_name_for_package("optimum-intel[openvino]>=1.25.1,<1.26") == "optimum.intel"


def test_package_name_strips_specifiers_and_extras():
    assert _package_name("optimum-intel[openvino]>=1.25.1,<1.26") == "optimum-intel"


def test_gpu_runtime_packages_include_sentencepiece():
    assert "sentencepiece" in _ort_runtime_packages()
    assert any(package.startswith("sentencepiece") for package in _openvino_runtime_packages())


def test_hf_runtime_packages_allow_newer_torch():
    assert "torch>=2.7" in _hf_runtime_packages()
    assert all(package != "torch>=2.7,<2.9" for package in _hf_runtime_packages())


def test_bootstrap_skips_when_modules_exist(monkeypatch):
    calls = []

    def fake_find_spec(name):
        return object()

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr("polylinguist.services.translation.importlib.util.find_spec", fake_find_spec)
    monkeypatch.setattr("polylinguist.services.translation.subprocess.run", fake_run)

    _ensure_python_packages(["huggingface-hub", "transformers"], "MarianMT")
    assert calls == []


def test_module_available_handles_missing_parent_package(monkeypatch):
    def fake_find_spec(name):
        if name == "optimum.intel":
            raise ModuleNotFoundError("No module named 'optimum'")
        return object()

    monkeypatch.setattr("polylinguist.services.translation.importlib.util.find_spec", fake_find_spec)

    assert _module_available("optimum.intel") is False


def test_package_runtime_ready_detects_incompatible_version(monkeypatch):
    monkeypatch.setattr("polylinguist.services.translation._module_available", lambda module_name: True)
    monkeypatch.setattr("polylinguist.services.translation.importlib_metadata.version", lambda package_name: "4.57.6")

    assert _package_runtime_ready("transformers>=4.53,<4.54") is False


def test_package_runtime_ready_accepts_newer_torch_for_hf_runtime(monkeypatch):
    monkeypatch.setattr("polylinguist.services.translation._module_available", lambda module_name: True)
    monkeypatch.setattr("polylinguist.services.translation.importlib_metadata.version", lambda package_name: "2.12.0")

    assert _package_runtime_ready("torch>=2.7") is True


def test_openvino_python_compatibility_rejects_windows_python_314(monkeypatch):
    monkeypatch.setattr("polylinguist.services.translation.platform.system", lambda: "Windows")
    version_info = type("VersionInfo", (), {"major": 3, "minor": 14, "micro": 1, "__ge__": lambda self, other: (3, 14, 1) >= other})()
    monkeypatch.setattr("polylinguist.services.translation.sys.version_info", version_info)

    try:
        _ensure_openvino_python_compatibility()
    except TranslationError as exc:
        assert "Python 3.13 or earlier" in str(exc)
    else:
        raise AssertionError("Expected TranslationError for Windows Python 3.14 OpenVINO.")


def test_load_openvino_seq2seq_class_falls_back_to_openvino_submodule(monkeypatch):
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "optimum.intel":
            raise ImportError("top-level import unavailable")
        if name == "optimum.intel.openvino":
            return type("Module", (), {"OVModelForSeq2SeqLM": object()})()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert _load_openvino_seq2seq_class() is not None


def test_discovery_includes_current_python():
    candidates = _discover_python_executables()
    assert any(candidate.lower() == sys.executable.lower() for candidate in candidates)


def test_resolve_runtime_prefers_existing_interpreter(monkeypatch):
    monkeypatch.setattr(
        "polylinguist.services.translation._discover_python_executables",
        lambda: [r"C:\\PythonA\\python.exe", r"C:\\PythonB\\python.exe"],
    )

    def fake_probe(executable, required_modules):
        if "PythonA" in executable:
            return None
        return type("Runtime", (), {
            "executable": executable,
            "missing_modules": (),
            "all_modules_present": True,
            "current": False,
        })()

    monkeypatch.setattr("polylinguist.services.translation._probe_python_runtime", fake_probe)
    runtime = _resolve_runtime(["transformers"])
    assert "PythonB" in runtime.executable
    assert runtime.all_modules_present is True


def test_device_preference_uses_cpu_when_requested(monkeypatch):
    fake_torch = type("Torch", (), {"cuda": type("Cuda", (), {"is_available": staticmethod(lambda: True)})()})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert _resolve_device_preference("cpu") == "cpu"


def test_device_preference_rejects_missing_cuda(monkeypatch):
    fake_torch = type("Torch", (), {"cuda": type("Cuda", (), {"is_available": staticmethod(lambda: False)})()})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    try:
        _resolve_device_preference("cuda")
    except TranslationError as exc:
        assert "CUDA is not available" in str(exc)
    else:
        raise AssertionError("Expected TranslationError when CUDA is unavailable.")


def test_device_preference_rejects_missing_directml(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", type("Ort", (), {"get_available_providers": staticmethod(lambda: ["CPUExecutionProvider"])})())
    try:
        _resolve_device_preference("directml")
    except TranslationError as exc:
        assert "DirectML is not available" in str(exc)
    else:
        raise AssertionError("Expected TranslationError when DirectML is unavailable.")


def test_huggingface_windows_cache_sets_no_symlink_mode(monkeypatch):
    monkeypatch.setattr("polylinguist.services.translation.platform.system", lambda: "Windows")
    monkeypatch.delenv("HF_HUB_DISABLE_SYMLINKS", raising=False)

    _configure_huggingface_windows_cache()

    assert os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"


def test_openvino_install_uses_compatible_worker_runtime(monkeypatch, tmp_path: Path):
    registry = InstalledModelRegistry(tmp_path / "installed.json")
    translator = MarianTranslator(registry, tmp_path / "models")

    fake_runtime = type(
        "Runtime",
        (),
        {
            "executable": r"C:\\Python313\\python.exe",
            "current": False,
            "has_cuda": False,
        },
    )()

    monkeypatch.setattr(
        "polylinguist.services.translation.resolve_runtime_for_target",
        lambda target, packages, prefer_cuda=False, system_name=None: fake_runtime,
    )
    monkeypatch.setattr(
        "polylinguist.services.translation._run_worker",
        lambda executable, command, payload, progress=None: (
            executable == r"C:\\Python313\\python.exe" and command == "install-marian-openvino" and "installed"
        ),
    )
    monkeypatch.setattr(
        "polylinguist.services.translation._runtime_metadata_snapshot",
        lambda runtime, packages: {"python_executable": runtime.executable, "python_version": "3.13.9"},
    )

    detail = translator.install(
        ModelDescriptor(
            provider="marian",
            model_id="Helsinki-NLP/opus-mt-en-ine",
            label="Fake Marian",
            source_lang="eng",
            target_lang="pol",
            size_mb=320,
            available=True,
            direct=False,
            installed=False,
            installed_targets=(),
            supported_targets=("cpu", "openvino_gpu"),
            recommended_target="openvino_gpu",
        ),
        device_preference="openvino_gpu",
    )

    assert detail == "installed"
    metadata = registry.metadata_for("marian", "Helsinki-NLP/opus-mt-en-ine", "openvino_gpu")
    assert metadata is not None
    assert metadata["python_executable"] == r"C:\\Python313\\python.exe"


def test_huggingface_windows_privilege_error_is_rewritten():
    exc = OSError("symlink failure")
    exc.winerror = 1314  # type: ignore[attr-defined]

    try:
        _wrap_huggingface_windows_privilege_error(exc)
    except TranslationError as translated:
        assert "WinError 1314" in str(translated)
        assert "no-symlink cache mode" in str(translated)
    else:
        raise AssertionError("Expected TranslationError for WinError 1314.")
