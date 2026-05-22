import os
import sys

from polylinguist.services.translation import (
    _discover_python_executables,
    _ensure_python_packages,
    _configure_huggingface_windows_cache,
    _module_name_for_package,
    _resolve_device_preference,
    _resolve_runtime,
    _wrap_huggingface_windows_privilege_error,
    TranslationError,
)


def test_module_name_mapping():
    assert _module_name_for_package("huggingface-hub") == "huggingface_hub"
    assert _module_name_for_package("sentencepiece") == "sentencepiece"
    assert _module_name_for_package("onnxruntime-directml") == "onnxruntime"
    assert _module_name_for_package("optimum-intel") == "optimum.intel"


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
