from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import os
from queue import Empty, Queue
import shutil
import subprocess
import sys
from threading import Thread
import time
from typing import Callable

from polylinguist.services.languages import get_language
from polylinguist.services.local_models import hf_model_cache_exists
from polylinguist.services.model_catalog import ModelDescriptor
from polylinguist.services.model_registry import InstalledModelRegistry


PACKAGE_MODULES = {
    "huggingface-hub": "huggingface_hub",
    "sentencepiece": "sentencepiece",
    "torch": "torch",
    "transformers": "transformers",
    "argostranslate": "argostranslate",
}


class TranslationError(RuntimeError):
    pass


ProgressCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class TranslationRequest:
    provider: str
    model_id: str
    source_lang: str
    target_lang: str
    device_preference: str = "auto"


@dataclass(frozen=True)
class PythonRuntime:
    executable: str
    missing_modules: tuple[str, ...]
    all_modules_present: bool
    current: bool
    has_cuda: bool = False


class TranslatorAdapter:
    provider: str

    def install(
        self,
        descriptor: ModelDescriptor,
        progress: ProgressCallback | None = None,
        device_preference: str = "auto",
    ) -> str:
        raise NotImplementedError

    def translate_batch(self, request: TranslationRequest, cues: list[str], progress: ProgressCallback | None = None) -> list[str]:
        raise NotImplementedError


class TranslationManager:
    def __init__(self, registry: InstalledModelRegistry) -> None:
        self.registry = registry
        self.adapters: dict[str, TranslatorAdapter] = {
            "argos": ArgosTranslator(registry),
            "marian": MarianTranslator(registry),
            "nllb": NllbTranslator(registry),
        }

    def install(
        self,
        descriptor: ModelDescriptor,
        progress: ProgressCallback | None = None,
        device_preference: str = "auto",
    ) -> str:
        adapter = self.adapters[descriptor.provider]
        return adapter.install(descriptor, progress, device_preference)

    def is_installed(self, provider: str, model_id: str) -> bool:
        if self.registry.is_installed(provider, model_id):
            return True
        if provider in {"marian", "nllb"}:
            return hf_model_cache_exists(model_id)
        return False

    def translate_batch(self, request: TranslationRequest, cues: list[str], progress: ProgressCallback | None = None) -> list[str]:
        if not self.is_installed(request.provider, request.model_id):
            raise TranslationError(f"Model {request.provider}:{request.model_id} is not installed.")
        adapter = self.adapters[request.provider]
        return adapter.translate_batch(request, cues, progress)


class ArgosTranslator(TranslatorAdapter):
    provider = "argos"

    def __init__(self, registry: InstalledModelRegistry) -> None:
        self.registry = registry

    def install(
        self,
        descriptor: ModelDescriptor,
        progress: ProgressCallback | None = None,
        device_preference: str = "auto",
    ) -> str:
        runtime = _resolve_runtime(["argostranslate"])
        _notify(progress, "runtime", f"Using Python runtime: {runtime.executable}")
        if runtime.current:
            _ensure_python_packages(["argostranslate"], "Argos Translate", progress)
            detail = _install_argos_current(descriptor, progress)
        else:
            detail = _run_worker(
                runtime.executable,
                "install-argos",
                {
                    "model_id": descriptor.model_id,
                    "install_strategy": descriptor.install_strategy,
                },
                progress=progress,
            )
        self.registry.mark_installed(
            self.provider,
            descriptor.model_id,
            {
                "python_executable": runtime.executable,
                "detail": detail,
            },
        )
        return detail

    def translate_batch(self, request: TranslationRequest, cues: list[str], progress: ProgressCallback | None = None) -> list[str]:
        metadata = self.registry.metadata_for(self.provider, request.model_id) or {}
        runtime_executable = metadata.get("python_executable", sys.executable)
        if request.device_preference == "cuda":
            _notify(progress, "runtime", "Argos Translate is CPU-only. Falling back to CPU execution.")
        if _same_executable(runtime_executable, sys.executable):
            _notify(progress, "runtime", "Using Argos in the active Python runtime.")
            _ensure_python_packages(["argostranslate"], "Argos Translate")
            _notify(progress, "translate", f"Translating {len(cues)} cues with Argos.")
            return _translate_argos_current(request, cues, progress)
        _notify(progress, "runtime", f"Using Argos worker runtime: {runtime_executable}")
        return _run_worker(
            runtime_executable,
            "translate-argos",
            {
                "model_id": request.model_id,
                "source_lang": request.source_lang,
                "target_lang": request.target_lang,
                "cues": cues,
            },
            progress=progress,
        )


class MarianTranslator(TranslatorAdapter):
    provider = "marian"

    def __init__(self, registry: InstalledModelRegistry) -> None:
        self.registry = registry
        self._bundles: dict[tuple[str, str], dict[str, object]] = {}

    def install(
        self,
        descriptor: ModelDescriptor,
        progress: ProgressCallback | None = None,
        device_preference: str = "auto",
    ) -> str:
        runtime = _resolve_runtime(_hf_runtime_packages(), prefer_cuda=device_preference == "cuda")
        _notify(progress, "runtime", f"Using Python runtime: {runtime.executable}")
        if runtime.current:
            _ensure_python_packages(_hf_runtime_packages(), "MarianMT", progress)
            detail = _install_hf_current(descriptor.model_id, progress)
        else:
            detail = _run_worker(
                runtime.executable,
                "install-hf",
                {
                    "model_id": descriptor.model_id,
                },
                progress=progress,
            )
        self.registry.mark_installed(
            self.provider,
            descriptor.model_id,
            {
                "python_executable": runtime.executable,
                "detail": detail,
            },
        )
        return detail

    def translate_batch(self, request: TranslationRequest, cues: list[str], progress: ProgressCallback | None = None) -> list[str]:
        metadata = self.registry.metadata_for(self.provider, request.model_id) or {}
        runtime_executable = metadata.get("python_executable", sys.executable)
        if request.device_preference == "cuda" and not _python_runtime_has_cuda(runtime_executable):
            preferred_runtime = _resolve_runtime(_hf_runtime_packages(), prefer_cuda=True)
            if preferred_runtime.has_cuda:
                runtime_executable = preferred_runtime.executable
        if _same_executable(runtime_executable, sys.executable):
            _notify(progress, "runtime", "Using MarianMT in the active Python runtime.")
            _ensure_python_packages(_hf_runtime_packages(), "MarianMT")
            return self._translate_current(
                request.model_id,
                request.target_lang,
                cues,
                request.device_preference,
                progress,
            )
        _notify(progress, "runtime", f"Using MarianMT worker runtime: {runtime_executable}")
        return _run_worker(
            runtime_executable,
            "translate-marian",
            {
                "model_id": request.model_id,
                "target_lang": request.target_lang,
                "cues": cues,
                "device_preference": request.device_preference,
            },
            progress=progress,
        )

    def _translate_current(
        self,
        model_id: str,
        target_lang: str,
        cues: list[str],
        device_preference: str,
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        device = _resolve_device_preference(device_preference, progress)
        bundle = self._get_bundle(model_id, device, progress)
        return _translate_seq2seq_bundle(bundle, _prepare_marian_batch(model_id, target_lang, cues), progress)

    def _get_bundle(self, model_id: str, device: str, progress: ProgressCallback | None = None) -> dict[str, object]:
        cache_key = (model_id, device)
        if cache_key in self._bundles:
            return self._bundles[cache_key]
        import torch  # type: ignore
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        _notify(progress, "load", f"Loading tokenizer for {model_id}.")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        _notify(progress, "load", f"Loading model weights for {model_id}.")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        if device != "cpu":
            model = model.to(device)
        bundle = {"tokenizer": tokenizer, "model": model, "device": device}
        self._bundles[cache_key] = bundle
        _notify(progress, "load", f"Model loaded on {device}.")
        return bundle


class NllbTranslator(TranslatorAdapter):
    provider = "nllb"

    def __init__(self, registry: InstalledModelRegistry) -> None:
        self.registry = registry
        self._bundles: dict[tuple[str, str], dict[str, object]] = {}

    def install(
        self,
        descriptor: ModelDescriptor,
        progress: ProgressCallback | None = None,
        device_preference: str = "auto",
    ) -> str:
        runtime = _resolve_runtime(_hf_runtime_packages(), prefer_cuda=device_preference == "cuda")
        _notify(progress, "runtime", f"Using Python runtime: {runtime.executable}")
        if runtime.current:
            _ensure_python_packages(_hf_runtime_packages(), "NLLB", progress)
            detail = _install_hf_current(descriptor.model_id, progress)
        else:
            detail = _run_worker(
                runtime.executable,
                "install-hf",
                {
                    "model_id": descriptor.model_id,
                },
                progress=progress,
            )
        self.registry.mark_installed(
            self.provider,
            descriptor.model_id,
            {
                "python_executable": runtime.executable,
                "detail": detail,
            },
        )
        return detail

    def translate_batch(self, request: TranslationRequest, cues: list[str], progress: ProgressCallback | None = None) -> list[str]:
        metadata = self.registry.metadata_for(self.provider, request.model_id) or {}
        runtime_executable = metadata.get("python_executable", sys.executable)
        if request.device_preference == "cuda" and not _python_runtime_has_cuda(runtime_executable):
            preferred_runtime = _resolve_runtime(_hf_runtime_packages(), prefer_cuda=True)
            if preferred_runtime.has_cuda:
                runtime_executable = preferred_runtime.executable
        if _same_executable(runtime_executable, sys.executable):
            _notify(progress, "runtime", "Using NLLB in the active Python runtime.")
            _ensure_python_packages(_hf_runtime_packages(), "NLLB")
            device = _resolve_device_preference(request.device_preference, progress)
            bundle = self._get_bundle(request.model_id, device, progress)
            source = get_language(request.source_lang)
            target = get_language(request.target_lang)
            if not source or not target or not source.nllb_code or not target.nllb_code:
                raise TranslationError("Requested language pair is not available in NLLB.")
            tokenizer = bundle["tokenizer"]
            model = bundle["model"]
            encoded = tokenizer(cues, return_tensors="pt", padding=True, truncation=True)
            if device != "cpu":
                encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target.nllb_code),
            )
            return tokenizer.batch_decode(generated, skip_special_tokens=True)
        _notify(progress, "runtime", f"Using NLLB worker runtime: {runtime_executable}")
        return _run_worker(
            runtime_executable,
            "translate-nllb",
            {
                "model_id": request.model_id,
                "source_lang": request.source_lang,
                "target_lang": request.target_lang,
                "cues": cues,
                "device_preference": request.device_preference,
            },
            progress=progress,
        )

    def _get_bundle(self, model_id: str, device: str, progress: ProgressCallback | None = None) -> dict[str, object]:
        cache_key = (model_id, device)
        if cache_key in self._bundles:
            return self._bundles[cache_key]
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

        _notify(progress, "load", f"Loading tokenizer for {model_id}.")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        _notify(progress, "load", f"Loading model weights for {model_id}.")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        if device != "cpu":
            model = model.to(device)
        bundle = {"tokenizer": tokenizer, "model": model, "device": device}
        self._bundles[cache_key] = bundle
        _notify(progress, "load", f"Model loaded on {device}.")
        return bundle


def _install_argos_current(descriptor: ModelDescriptor, progress: ProgressCallback | None = None) -> str:
    import argostranslate.package  # type: ignore

    source, target = _argos_path_from_model_id(descriptor.model_id)
    package_specs = [(source, target)] if descriptor.install_strategy == "direct" else [(source, "en"), ("en", target)]
    _notify(progress, "index", "Refreshing Argos package index.")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    installed_pairs: list[str] = []

    for from_code, to_code in package_specs:
        _notify(progress, "download", f"Downloading Argos package {from_code}->{to_code}.")
        match = next(
            (pkg for pkg in available if pkg.from_code == from_code and pkg.to_code == to_code),
            None,
        )
        if not match:
            raise TranslationError(f"Missing Argos package for {from_code}->{to_code}.")
        package_path = match.download()
        _notify(progress, "install", f"Installing Argos package {from_code}->{to_code}.")
        argostranslate.package.install_from_path(package_path)
        installed_pairs.append(f"{from_code}->{to_code}")
    return f"Installed Argos packages: {', '.join(installed_pairs)}"


def _translate_argos_current(
    request: TranslationRequest,
    cues: list[str],
    progress: ProgressCallback | None = None,
) -> list[str]:
    import argostranslate.translate  # type: ignore

    source, target = _argos_path_from_model_id(request.model_id)
    translations: list[str] = []
    total = len(cues)
    for index, text in enumerate(cues, start=1):
        _notify(progress, "translate", f"Translating cue {index}/{total}.")
        if "+en-" in request.model_id:
            translated = argostranslate.translate.translate(
                argostranslate.translate.translate(text, source, "en"),
                "en",
                target,
            )
        else:
            translated = argostranslate.translate.translate(text, source, target)
        translations.append(translated)
    return translations


def _install_hf_current(model_id: str, progress: ProgressCallback | None = None) -> str:
    from huggingface_hub import snapshot_download  # type: ignore

    _notify(progress, "download", f"Downloading model weights for {model_id}.")
    location = snapshot_download(repo_id=model_id)
    return f"Downloaded {model_id} to {location}"


def _translate_seq2seq_bundle(
    bundle: dict[str, object],
    cues: list[str],
    progress: ProgressCallback | None = None,
) -> list[str]:
    if not cues:
        return []
    import torch  # type: ignore

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = str(bundle.get("device") or "cpu")
    batch_size = 8
    translations: list[str] = []
    total_batches = (len(cues) + batch_size - 1) // batch_size
    model.eval()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(cues), batch_size), start=1):
            batch = cues[start : start + batch_size]
            _notify(progress, "translate", f"Translating batch {batch_index}/{total_batches}.")
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            if device != "cpu":
                encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(**encoded, max_new_tokens=256)
            translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return translations


def _prepare_marian_batch(model_id: str, target_lang: str, cues: list[str]) -> list[str]:
    target = get_language(target_lang)
    if target is None:
        return cues
    token: str | None = None
    if "opus-mt-en-ine" in model_id:
        token = {
            "pol": "pol",
            "por": "por",
            "pob": "por",
            "rus": "rus",
            "ukr": "ukr",
            "cze": "ces",
            "ger": "deu",
            "dut": "nld",
            "swe": "swe",
            "hin": "hin",
        }.get(target.canonical)
    elif "opus-mt-en-ROMANCE" in model_id:
        token = {
            "fre": "fr",
            "spa": "es",
            "ita": "it",
            "por": "pt",
            "pob": "pt_BR",
        }.get(target.canonical)
    elif "opus-mt-en-trk" in model_id:
        token = {
            "tur": "tur",
        }.get(target.canonical)
    if not token:
        return cues
    return [f">>{token}<< {cue}" for cue in cues]


def _resolve_device_preference(device_preference: str, progress: ProgressCallback | None = None) -> str:
    import torch  # type: ignore

    normalized = (device_preference or "auto").lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise TranslationError("GPU processing was requested, but CUDA is not available in the selected Python runtime.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_runtime(packages: list[str], prefer_cuda: bool = False) -> PythonRuntime:
    required_modules = tuple(_module_name_for_package(package) for package in packages)
    best_match: PythonRuntime | None = None
    for executable in _discover_python_executables():
        runtime = _probe_python_runtime(executable, required_modules)
        if not runtime or not runtime.all_modules_present:
            continue
        if prefer_cuda and runtime.has_cuda:
            return runtime
        if best_match is None:
            best_match = runtime
    current = _probe_current_runtime(required_modules)
    if prefer_cuda and current.has_cuda and current.all_modules_present:
        return current
    if best_match is not None:
        return best_match
    return current


def _probe_current_runtime(required_modules: tuple[str, ...]) -> PythonRuntime:
    missing = tuple(
        module_name
        for module_name in required_modules
        if importlib.util.find_spec(module_name) is None
    )
    return PythonRuntime(
        executable=sys.executable,
        missing_modules=missing,
        all_modules_present=not missing,
        current=True,
        has_cuda=_current_runtime_has_cuda(),
    )


def _discover_python_executables() -> list[str]:
    candidates: list[str] = [sys.executable]
    env_override = os.getenv("POLYLINGUIST_PYTHON")
    if env_override:
        candidates.append(env_override)
    candidates.extend(_discover_local_virtualenvs())

    candidates.extend(_discover_with_command(["py", "-0p"]))
    candidates.extend(_discover_with_command(["where", "python"]))
    candidates.extend(_discover_with_command(["where", "py"]))

    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.normcase(os.path.abspath(candidate.strip()))
        if normalized in seen:
            continue
        if os.path.exists(normalized):
            deduped.append(normalized)
            seen.add(normalized)
    return deduped


def _discover_local_virtualenvs() -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    candidates: list[str] = []
    for name in [".venv", ".benchmarks\\venv", "venv"]:
        executable = repo_root / name / "Scripts" / "python.exe"
        if executable.exists():
            candidates.append(str(executable))
    return candidates


def _discover_with_command(command: list[str]) -> list[str]:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return []
    paths: list[str] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-V:") and "*" in stripped:
            stripped = stripped.split("*", 1)[1].strip()
        if stripped.endswith("*"):
            stripped = stripped[:-1].strip()
        if os.path.exists(stripped):
            paths.append(stripped)
    return paths


def _probe_python_runtime(executable: str, required_modules: tuple[str, ...]) -> PythonRuntime | None:
    if _same_executable(executable, sys.executable):
        return _probe_current_runtime(required_modules)
    probe_code = (
        "import importlib.util,json,sys;"
        "mods=sys.argv[1:];"
        "missing=[m for m in mods if importlib.util.find_spec(m) is None];"
        "torch_spec=importlib.util.find_spec('torch');"
        "has_cuda=bool(__import__('torch').cuda.is_available()) if torch_spec is not None else False;"
        "print(json.dumps({'missing':missing,'has_cuda':has_cuda}))"
    )
    try:
        completed = subprocess.run(
            [executable, "-c", probe_code, *required_modules],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        payload = json.loads(completed.stdout.strip() or "{}")
    except Exception:
        return None
    missing = tuple(payload.get("missing", []))
    return PythonRuntime(
        executable=executable,
        missing_modules=missing,
        all_modules_present=not missing,
        current=False,
        has_cuda=bool(payload.get("has_cuda")),
    )


def _ensure_python_packages(packages: list[str], feature_name: str, progress: ProgressCallback | None = None) -> None:
    missing = [package for package in packages if importlib.util.find_spec(_module_name_for_package(package)) is None]
    if not missing:
        _notify(progress, "runtime", f"{feature_name} runtime already available in the active interpreter.")
        return
    _notify(progress, "runtime", f"Installing Python packages for {feature_name}: {', '.join(missing)}")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", *missing],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
                _notify(progress, "runtime", stripped)
        return_code = process.wait()
        if return_code == 0:
            return
        detail = lines[-1] if lines else f"pip exited with code {return_code}"
        raise TranslationError(
            f"Polylinguist failed to install the Python runtime packages for {feature_name}: {detail}"
        )
    except OSError as exc:
        raise TranslationError(
            f"Polylinguist could not launch pip for {feature_name}: {exc}"
        ) from exc


def _run_worker(
    executable: str,
    command: str,
    payload: dict[str, object],
    progress: ProgressCallback | None = None,
) -> object:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    worker_module = "polylinguist.model_worker"
    try:
        process = subprocess.Popen(
            [executable, "-m", worker_module, command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
            cwd=str(repo_root),
            env=env,
        )
        assert process.stdin is not None
        process.stdin.write(json.dumps(payload))
        process.stdin.close()
    except OSError as exc:
        raise TranslationError(f"Polylinguist could not launch worker Python: {exc}") from exc

    result: object | None = None
    lines: list[str] = []
    queue: Queue[str | None] = Queue()

    def pump_stdout() -> None:
        assert process.stdout is not None
        for raw_line in process.stdout:
            queue.put(raw_line)
        queue.put(None)

    reader = Thread(target=pump_stdout, daemon=True)
    reader.start()

    deadline = time.monotonic() + 3600
    finished = False
    while not finished:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            raise TranslationError("Polylinguist worker timed out.")
        try:
            line = queue.get(timeout=min(0.5, remaining))
        except Empty:
            if process.poll() is not None and queue.empty():
                break
            continue
        if line is None:
            finished = True
            continue
        stripped = line.strip()
        if not stripped:
            continue
        lines.append(stripped)
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            _notify(progress, "worker", stripped)
            continue
        if message.get("event") == "progress":
            _notify(progress, str(message.get("stage") or "worker"), str(message.get("message") or ""))
        elif message.get("event") == "result":
            if not message.get("ok"):
                raise TranslationError(str(message.get("error") or "Polylinguist worker failed."))
            result = message.get("result")
        else:
            _notify(progress, "worker", stripped)

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise TranslationError("Polylinguist worker timed out during shutdown.") from exc

    if process.returncode != 0:
        detail = lines[-1].strip() if lines else f"worker exited with code {process.returncode}"
        raise TranslationError(f"Polylinguist worker failed: {detail}")
    if result is None:
        raise TranslationError("Polylinguist worker returned no result.")
    return result


def _same_executable(left: str, right: str) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def _current_runtime_has_cuda() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _python_runtime_has_cuda(executable: str) -> bool:
    if _same_executable(executable, sys.executable):
        return _current_runtime_has_cuda()
    runtime = _probe_python_runtime(executable, ())
    return bool(runtime and runtime.has_cuda)


def _argos_path_from_model_id(model_id: str) -> tuple[str, str]:
    payload = model_id.replace("argos:", "", 1)
    if "+en-" in payload:
        left, right = payload.split("+")
        source = left.split("-")[0]
        target = right.split("-")[1]
        return source, target
    source, target = payload.split("-")
    return source, target


def _hf_runtime_packages() -> list[str]:
    return ["huggingface-hub", "transformers", "torch", "sentencepiece"]


def _module_name_for_package(package: str) -> str:
    return PACKAGE_MODULES.get(package, package.replace("-", "_"))


def _notify(progress: ProgressCallback | None, stage: str, message: str) -> None:
    if progress is not None:
        progress(stage, message)
