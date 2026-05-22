import sys

from polylinguist.services.translation import (
    _discover_python_executables,
    _ensure_python_packages,
    _module_name_for_package,
    _resolve_device_preference,
    _resolve_runtime,
    TranslationError,
)


def test_module_name_mapping():
    assert _module_name_for_package("huggingface-hub") == "huggingface_hub"
    assert _module_name_for_package("sentencepiece") == "sentencepiece"


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
