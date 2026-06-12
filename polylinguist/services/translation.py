from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
from importlib import metadata as importlib_metadata
import json
import os
import platform
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock, Thread
import time
from typing import Callable

from packaging.requirements import Requirement

from polylinguist.services.compatibility import (
    provider_target_block_reason,
    runtime_target_block_reason,
    system_target_block_reason,
)
from polylinguist.services.languages import get_language
from polylinguist.services.local_models import (
    hf_model_cache_exists,
    local_model_artifact_dir,
    local_model_artifact_exists,
)
from polylinguist.services.model_catalog import ModelDescriptor
from polylinguist.services.model_registry import InstalledModelRegistry


PACKAGE_MODULES = {
    "argostranslate": "argostranslate",
    "huggingface-hub": "huggingface_hub",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "onnxruntime-directml": "onnxruntime",
    "openvino": "openvino",
    "optimum": "optimum.onnxruntime",
    "optimum-intel": "optimum.intel",
    "pillow": "PIL",
    "sentencepiece": "sentencepiece",
    "torch": "torch",
    "transformers": "transformers",
}

MARIAN_BATCH_SIZE = 8
OPENVINO_MARIAN_BATCH_SIZE = 1
NLLB_BATCH_SIZE = 16
# Subtitle cues are short; keeping the NLLB decode cap tight avoids long stalls
# when a malformed cue never emits EOS and holds the whole batch open.
NLLB_MAX_NEW_TOKENS = 64


def _configure_huggingface_windows_cache() -> None:
    if platform.system().lower() != "windows":
        return
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")


def _wrap_huggingface_windows_privilege_error(exc: OSError) -> TranslationError:
    if getattr(exc, "winerror", None) != 1314:
        raise exc
    raise TranslationError(
        "Windows blocked Hugging Face cache symlink creation (WinError 1314). "
        "Polylinguist now uses Hugging Face no-symlink cache mode on Windows; retry the install. "
        "If the machine still blocks it, enable Windows Developer Mode or run Polylinguist once as administrator."
    ) from exc


_configure_huggingface_windows_cache()


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
    python_version: str | None = None


class TranslatorAdapter:
    provider: str

    def install(
        self,
        descriptor: ModelDescriptor,
        progress: ProgressCallback | None = None,
        device_preference: str = "auto",
    ) -> str:
        raise NotImplementedError

    def translate_batch(
        self,
        request: TranslationRequest,
        cues: list[str],
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        raise NotImplementedError

    def forget_model(self, model_id: str, target: str | None = None) -> None:
        return None

    def clear_runtime_state(self) -> None:
        return None


class TranslationManager:
    def __init__(self, registry: InstalledModelRegistry, model_artifacts_dir: Path) -> None:
        self.registry = registry
        self.model_artifacts_dir = model_artifacts_dir
        self.adapters: dict[str, TranslatorAdapter] = {
            "argos": ArgosTranslator(registry),
            "marian": MarianTranslator(registry, model_artifacts_dir),
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

    def is_installed(self, provider: str, model_id: str, device_preference: str = "auto") -> bool:
        normalized = _normalize_device_preference(device_preference)
        if provider == "argos":
            if self.registry.is_removed(provider, model_id):
                return False
            return self.registry.is_installed(provider, model_id)
        if provider == "marian":
            if normalized == "auto":
                return (
                    (not self.registry.is_removed(provider, model_id) and (
                        self.registry.is_installed(provider, model_id) or hf_model_cache_exists(model_id)
                    ))
                    or (
                        not self.registry.is_removed(provider, model_id, "directml")
                        and (
                            self.registry.is_installed(provider, model_id, "directml")
                            or local_model_artifact_exists(self.model_artifacts_dir, provider, model_id, "directml")
                        )
                    )
                    or (
                        not self.registry.is_removed(provider, model_id, "openvino_gpu")
                        and (
                            self.registry.is_installed(provider, model_id, "openvino_gpu")
                            or local_model_artifact_exists(self.model_artifacts_dir, provider, model_id, "openvino_gpu")
                        )
                    )
                )
            if normalized in {"directml", "openvino_gpu"}:
                if self.registry.is_removed(provider, model_id, normalized):
                    return False
                return self.registry.is_installed(provider, model_id, normalized) or local_model_artifact_exists(
                    self.model_artifacts_dir, provider, model_id, normalized
                )
            if self.registry.is_removed(provider, model_id):
                return False
            return self.registry.is_installed(provider, model_id) or hf_model_cache_exists(model_id)
        if provider == "nllb":
            if normalized in {"directml", "openvino_gpu"}:
                return False
            if self.registry.is_removed(provider, model_id):
                return False
            return self.registry.is_installed(provider, model_id) or hf_model_cache_exists(model_id)
        return self.registry.is_installed(provider, model_id)

    def translate_batch(
        self,
        request: TranslationRequest,
        cues: list[str],
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        if not self.is_installed(request.provider, request.model_id, request.device_preference):
            raise TranslationError(
                f"Model {request.provider}:{request.model_id} is not installed for {request.device_preference}."
            )
        adapter = self.adapters[request.provider]
        return adapter.translate_batch(request, cues, progress)

    def forget_model(self, provider: str, model_id: str, target: str | None = None) -> None:
        adapter = self.adapters.get(provider)
        if adapter is not None:
            adapter.forget_model(model_id, target)

    def clear_runtime_state(self) -> None:
        for adapter in self.adapters.values():
            adapter.clear_runtime_state()


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
        target = _normalize_device_preference(device_preference)
        provider_reason = provider_target_block_reason(self.provider, target)
        if provider_reason:
            raise TranslationError(provider_reason)
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
                "runtime": _runtime_metadata_snapshot(runtime, ["argostranslate"]),
                "detail": detail,
            },
        )
        return detail

    def translate_batch(
        self,
        request: TranslationRequest,
        cues: list[str],
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        metadata = self.registry.metadata_for(self.provider, request.model_id) or {}
        runtime_executable = metadata.get("python_executable", sys.executable)
        if request.device_preference not in {"auto", "cpu"}:
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

    def forget_model(self, model_id: str, target: str | None = None) -> None:
        return None


class MarianTranslator(TranslatorAdapter):
    provider = "marian"

    def __init__(self, registry: InstalledModelRegistry, model_artifacts_dir: Path) -> None:
        self.registry = registry
        self.model_artifacts_dir = model_artifacts_dir
        self._bundles: dict[tuple[str, str], dict[str, object]] = {}

    def install(
        self,
        descriptor: ModelDescriptor,
        progress: ProgressCallback | None = None,
        device_preference: str = "auto",
    ) -> str:
        target = _normalize_device_preference(device_preference)
        provider_reason = provider_target_block_reason(self.provider, target)
        if provider_reason:
            raise TranslationError(provider_reason)
        if target == "auto":
            target = descriptor.recommended_target or "cpu"
        if target == "directml":
            runtime = _resolve_runtime(_ort_runtime_packages())
            artifact_dir = self._artifact_dir(descriptor.model_id, "directml")
            _notify(progress, "runtime", f"Using Python runtime: {runtime.executable}")
            if runtime.current:
                _ensure_python_packages(_ort_runtime_packages(), "MarianMT DirectML", progress)
                detail = _install_marian_directml_current(descriptor.model_id, artifact_dir, progress)
            else:
                detail = _run_worker(
                    runtime.executable,
                    "install-marian-directml",
                    {"model_id": descriptor.model_id, "artifact_dir": str(artifact_dir)},
                    progress=progress,
                )
            self.registry.mark_installed(
                self.provider,
                descriptor.model_id,
                {
                    "python_executable": runtime.executable,
                    "runtime": _runtime_metadata_snapshot(runtime, _ort_runtime_packages()),
                    "detail": detail,
                    "artifact_dir": str(artifact_dir),
                },
                target="directml",
            )
            return detail

        if target == "openvino_gpu":
            runtime = _resolve_runtime(_openvino_runtime_packages())
            artifact_dir = self._artifact_dir(descriptor.model_id, "openvino_gpu")
            _notify(progress, "runtime", f"Using Python runtime: {runtime.executable}")
            if runtime.current:
                _ensure_python_packages(_openvino_runtime_packages(), "MarianMT OpenVINO", progress)
                detail = _install_marian_openvino_current(descriptor.model_id, artifact_dir, progress)
            else:
                detail = _run_worker(
                    runtime.executable,
                    "install-marian-openvino",
                    {"model_id": descriptor.model_id, "artifact_dir": str(artifact_dir)},
                    progress=progress,
                )
            self.registry.mark_installed(
                self.provider,
                descriptor.model_id,
                {
                    "python_executable": runtime.executable,
                    "runtime": _runtime_metadata_snapshot(runtime, _openvino_runtime_packages()),
                    "detail": detail,
                    "artifact_dir": str(artifact_dir),
                },
                target="openvino_gpu",
            )
            return detail

        runtime = _resolve_runtime(_hf_runtime_packages(), prefer_cuda=target == "cuda")
        _notify(progress, "runtime", f"Using Python runtime: {runtime.executable}")
        if runtime.current:
            _ensure_python_packages(_hf_runtime_packages(), "MarianMT", progress)
            detail = _install_hf_current(descriptor.model_id, progress)
        else:
            detail = _run_worker(
                runtime.executable,
                "install-hf",
                {"model_id": descriptor.model_id},
                progress=progress,
            )
        self.registry.mark_installed(
            self.provider,
            descriptor.model_id,
            {
                "python_executable": runtime.executable,
                "runtime": _runtime_metadata_snapshot(runtime, _hf_runtime_packages()),
                "detail": detail,
            },
        )
        return detail

    def translate_batch(
        self,
        request: TranslationRequest,
        cues: list[str],
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        target = self._effective_target(request.device_preference, request.model_id)
        if target == "directml":
            return self._translate_directml(request, cues, progress)
        if target == "openvino_gpu":
            return self._translate_openvino(request, cues, progress)
        return self._translate_torch(request, cues, progress)

    def _translate_torch(
        self,
        request: TranslationRequest,
        cues: list[str],
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        metadata = self.registry.metadata_for(self.provider, request.model_id) or {}
        runtime_executable = metadata.get("python_executable", sys.executable)
        if request.device_preference == "cuda" and not _python_runtime_has_cuda(runtime_executable):
            preferred_runtime = _resolve_runtime(_hf_runtime_packages(), prefer_cuda=True)
            if preferred_runtime.has_cuda:
                runtime_executable = preferred_runtime.executable
        if _same_executable(runtime_executable, sys.executable):
            _notify(progress, "runtime", "Using MarianMT in the active Python runtime.")
            _ensure_python_packages(_hf_runtime_packages(), "MarianMT")
            normalized = _normalize_device_preference(request.device_preference)
            device = _resolve_device_preference("cuda", progress) if normalized == "cuda" else "cpu"
            if normalized == "auto" and _current_runtime_has_cuda():
                device = "cuda"
            bundle = self._get_torch_bundle(request.model_id, device, progress)
            prepared = _prepare_marian_batch(request.model_id, request.target_lang, cues)
            return _translate_torch_seq2seq_bundle(bundle, prepared, progress)
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

    def _translate_directml(
        self,
        request: TranslationRequest,
        cues: list[str],
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        metadata = self.registry.metadata_for(self.provider, request.model_id, "directml") or {}
        artifact_dir = Path(metadata.get("artifact_dir") or self._artifact_dir(request.model_id, "directml"))
        runtime_executable = metadata.get("python_executable", sys.executable)
        if not artifact_dir.exists():
            raise TranslationError("Marian DirectML artifacts are not installed for this model.")
        if _same_executable(runtime_executable, sys.executable):
            _notify(progress, "runtime", "Using MarianMT DirectML in the active Python runtime.")
            _ensure_python_packages(_ort_runtime_packages(), "MarianMT DirectML")
            bundle = self._get_directml_bundle(artifact_dir, progress)
            prepared = _prepare_marian_batch(request.model_id, request.target_lang, cues)
            return _translate_ort_seq2seq_bundle(bundle, prepared, progress)
        _notify(progress, "runtime", f"Using MarianMT DirectML worker runtime: {runtime_executable}")
        return _run_worker(
            runtime_executable,
            "translate-marian-directml",
            {
                "artifact_dir": str(artifact_dir),
                "model_id": request.model_id,
                "target_lang": request.target_lang,
                "cues": cues,
            },
            progress=progress,
        )

    def _translate_openvino(
        self,
        request: TranslationRequest,
        cues: list[str],
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        metadata = self.registry.metadata_for(self.provider, request.model_id, "openvino_gpu") or {}
        artifact_dir = Path(metadata.get("artifact_dir") or self._artifact_dir(request.model_id, "openvino_gpu"))
        runtime_executable = metadata.get("python_executable", sys.executable)
        if not artifact_dir.exists():
            raise TranslationError("Marian OpenVINO artifacts are not installed for this model.")
        if _same_executable(runtime_executable, sys.executable):
            _notify(progress, "runtime", "Using MarianMT OpenVINO GPU in the active Python runtime.")
            _ensure_python_packages(_openvino_runtime_packages(), "MarianMT OpenVINO")
            bundle = self._get_openvino_bundle(artifact_dir, progress)
            prepared = _prepare_marian_batch(request.model_id, request.target_lang, cues)
            return _translate_openvino_seq2seq_bundle(bundle, prepared, progress)
        _notify(progress, "runtime", f"Using MarianMT OpenVINO worker runtime: {runtime_executable}")
        return _run_worker(
            runtime_executable,
            "translate-marian-openvino",
            {
                "artifact_dir": str(artifact_dir),
                "model_id": request.model_id,
                "target_lang": request.target_lang,
                "cues": cues,
            },
            progress=progress,
        )

    def _artifact_dir(self, model_id: str, target: str) -> Path:
        return local_model_artifact_dir(self.model_artifacts_dir, self.provider, model_id, target)

    def forget_model(self, model_id: str, target: str | None = None) -> None:
        normalized = _normalize_device_preference(target or "auto")
        if normalized in {"directml", "openvino_gpu"}:
            artifact_dir = str(self._artifact_dir(model_id, normalized))
            for cache_key in list(self._bundles):
                if cache_key == (artifact_dir, normalized):
                    self._bundles.pop(cache_key, None)
            return
        for cache_key in list(self._bundles):
            cache_model_id, cache_device = cache_key
            if cache_model_id == model_id and cache_device in {"cpu", "cuda"}:
                self._bundles.pop(cache_key, None)

    def clear_runtime_state(self) -> None:
        self._bundles.clear()

    def _effective_target(self, requested_target: str, model_id: str) -> str:
        normalized = _normalize_device_preference(requested_target)
        if normalized != "auto":
            return normalized
        if _current_runtime_has_cuda():
            return "cuda"
        if self.registry.is_installed(self.provider, model_id, "openvino_gpu") or local_model_artifact_exists(
            self.model_artifacts_dir, self.provider, model_id, "openvino_gpu"
        ):
            return "openvino_gpu"
        if self.registry.is_installed(self.provider, model_id, "directml") or local_model_artifact_exists(
            self.model_artifacts_dir, self.provider, model_id, "directml"
        ):
            return "directml"
        return "cpu"

    def _get_torch_bundle(
        self,
        model_id: str,
        device: str,
        progress: ProgressCallback | None = None,
    ) -> dict[str, object]:
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
        if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
            model.generation_config.max_length = None
        bundle = {"tokenizer": tokenizer, "model": model, "device": device, "lock": Lock()}
        self._bundles[cache_key] = bundle
        _notify(progress, "load", f"Model loaded on {device}.")
        return bundle

    def _get_directml_bundle(
        self,
        artifact_dir: Path,
        progress: ProgressCallback | None = None,
    ) -> dict[str, object]:
        cache_key = (str(artifact_dir), "directml")
        if cache_key in self._bundles:
            return self._bundles[cache_key]
        from optimum.onnxruntime import ORTModelForSeq2SeqLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        _notify(progress, "load", f"Loading Marian DirectML artifacts from {artifact_dir}.")
        tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir))
        model = ORTModelForSeq2SeqLM.from_pretrained(str(artifact_dir), provider="DmlExecutionProvider")
        bundle = {"tokenizer": tokenizer, "model": model, "device": "directml", "lock": Lock()}
        self._bundles[cache_key] = bundle
        _notify(progress, "load", "DirectML session is ready.")
        return bundle

    def _get_openvino_bundle(
        self,
        artifact_dir: Path,
        progress: ProgressCallback | None = None,
    ) -> dict[str, object]:
        cache_key = (str(artifact_dir), "openvino_gpu")
        if cache_key in self._bundles:
            return self._bundles[cache_key]
        from optimum.intel import OVModelForSeq2SeqLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        _notify(progress, "load", f"Loading Marian OpenVINO artifacts from {artifact_dir}.")
        tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir))
        model = OVModelForSeq2SeqLM.from_pretrained(
            str(artifact_dir),
            device="gpu",
            compile=False,
            ov_config={"CACHE_DIR": str(artifact_dir / "model_cache")},
        )
        _notify(progress, "compile", "Compiling OpenVINO GPU graph.")
        model.to("gpu")
        model.compile()
        bundle = {"tokenizer": tokenizer, "model": model, "device": "openvino_gpu", "lock": Lock()}
        self._bundles[cache_key] = bundle
        _notify(progress, "load", "OpenVINO GPU model is ready.")
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
        target = _normalize_device_preference(device_preference)
        provider_reason = provider_target_block_reason(self.provider, target)
        if provider_reason:
            raise TranslationError(provider_reason)
        runtime = _resolve_runtime(_hf_runtime_packages(), prefer_cuda=target == "cuda")
        _notify(progress, "runtime", f"Using Python runtime: {runtime.executable}")
        if runtime.current:
            _ensure_python_packages(_hf_runtime_packages(), "NLLB", progress)
            detail = _install_hf_current(descriptor.model_id, progress)
        else:
            detail = _run_worker(
                runtime.executable,
                "install-hf",
                {"model_id": descriptor.model_id},
                progress=progress,
            )
        self.registry.mark_installed(
            self.provider,
            descriptor.model_id,
            {
                "python_executable": runtime.executable,
                "runtime": _runtime_metadata_snapshot(runtime, _hf_runtime_packages()),
                "detail": detail,
            },
        )
        return detail

    def translate_batch(
        self,
        request: TranslationRequest,
        cues: list[str],
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        target = _normalize_device_preference(request.device_preference)
        provider_reason = provider_target_block_reason(self.provider, target)
        if provider_reason:
            raise TranslationError(provider_reason)
        metadata = self.registry.metadata_for(self.provider, request.model_id) or {}
        runtime_executable = metadata.get("python_executable", sys.executable)
        if request.device_preference == "cuda" and not _python_runtime_has_cuda(runtime_executable):
            preferred_runtime = _resolve_runtime(_hf_runtime_packages(), prefer_cuda=True)
            if preferred_runtime.has_cuda:
                runtime_executable = preferred_runtime.executable
        if _same_executable(runtime_executable, sys.executable):
            _notify(progress, "runtime", "Using NLLB in the active Python runtime.")
            _ensure_python_packages(_hf_runtime_packages(), "NLLB")
            normalized = _normalize_device_preference(request.device_preference)
            device = _resolve_device_preference("cuda", progress) if normalized == "cuda" else "cpu"
            if normalized == "auto" and _current_runtime_has_cuda():
                device = "cuda"
            bundle = self._get_bundle(request.model_id, device, progress)
            source = get_language(request.source_lang)
            target_lang = get_language(request.target_lang)
            if not source or not target_lang or not source.nllb_code or not target_lang.nllb_code:
                raise TranslationError("Requested language pair is not available in NLLB.")
            tokenizer = bundle["tokenizer"]
            model = bundle["model"]
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang.nllb_code)
            import torch  # type: ignore

            translations: list[str] = []
            model.eval()
            lock: Lock = bundle["lock"]  # type: ignore[assignment]
            with lock:
                with torch.no_grad():
                    for batch_index, start in enumerate(range(0, len(cues), NLLB_BATCH_SIZE), start=1):
                        _notify(
                            progress,
                            "translate",
                            f"Translating batch {batch_index}/{max((len(cues) + NLLB_BATCH_SIZE - 1) // NLLB_BATCH_SIZE, 1)}.",
                        )
                        batch = cues[start : start + NLLB_BATCH_SIZE]
                        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                        if device != "cpu":
                            encoded = {key: value.to(device) for key, value in encoded.items()}
                        generated = model.generate(
                            **encoded,
                            forced_bos_token_id=forced_bos_token_id,
                            max_new_tokens=NLLB_MAX_NEW_TOKENS,
                        )
                        translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            return translations
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
        if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
            model.generation_config.max_length = None
        bundle = {"tokenizer": tokenizer, "model": model, "device": device, "lock": Lock()}
        self._bundles[cache_key] = bundle
        _notify(progress, "load", f"Model loaded on {device}.")
        return bundle

    def forget_model(self, model_id: str, target: str | None = None) -> None:
        for cache_key in list(self._bundles):
            cache_model_id, cache_device = cache_key
            if cache_model_id != model_id:
                continue
            if target is None:
                self._bundles.pop(cache_key, None)
                continue
            normalized = _normalize_device_preference(target)
            if normalized == "auto" or normalized == cache_device:
                self._bundles.pop(cache_key, None)

    def clear_runtime_state(self) -> None:
        self._bundles.clear()


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

    _configure_huggingface_windows_cache()
    _notify(progress, "download", f"Downloading model weights for {model_id}.")
    if os.environ.get("HF_HUB_DISABLE_SYMLINKS") == "1" and platform.system().lower() == "windows":
        _notify(progress, "runtime", "Windows detected; using Hugging Face no-symlink cache mode.")
    try:
        location = snapshot_download(repo_id=model_id)
    except OSError as exc:
        _wrap_huggingface_windows_privilege_error(exc)
    return f"Downloaded {model_id} to {location}"


def _install_marian_directml_current(
    model_id: str,
    artifact_dir: Path,
    progress: ProgressCallback | None = None,
) -> str:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    artifact_dir.mkdir(parents=True, exist_ok=True)
    _notify(progress, "download", f"Exporting MarianMT model to DirectML-ready ONNX artifacts for {model_id}.")
    _configure_huggingface_windows_cache()
    if os.environ.get("HF_HUB_DISABLE_SYMLINKS") == "1" and platform.system().lower() == "windows":
        _notify(progress, "runtime", "Windows detected; using Hugging Face no-symlink cache mode.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)
    except OSError as exc:
        _wrap_huggingface_windows_privilege_error(exc)
    model.save_pretrained(str(artifact_dir))
    tokenizer.save_pretrained(str(artifact_dir))
    return f"Exported DirectML artifacts for {model_id} to {artifact_dir}"


def _install_marian_openvino_current(
    model_id: str,
    artifact_dir: Path,
    progress: ProgressCallback | None = None,
) -> str:
    from transformers import AutoTokenizer  # type: ignore

    _ensure_openvino_python_compatibility()
    OVModelForSeq2SeqLM = _load_openvino_seq2seq_class()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _notify(progress, "download", f"Exporting MarianMT model to OpenVINO artifacts for {model_id}.")
    _configure_huggingface_windows_cache()
    if os.environ.get("HF_HUB_DISABLE_SYMLINKS") == "1" and platform.system().lower() == "windows":
        _notify(progress, "runtime", "Windows detected; using Hugging Face no-symlink cache mode.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, compile=False)
    except OSError as exc:
        _wrap_huggingface_windows_privilege_error(exc)
    model.save_pretrained(str(artifact_dir))
    tokenizer.save_pretrained(str(artifact_dir))
    return f"Exported OpenVINO artifacts for {model_id} to {artifact_dir}"


def _translate_torch_seq2seq_bundle(
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
    total_batches = max((len(cues) + MARIAN_BATCH_SIZE - 1) // MARIAN_BATCH_SIZE, 1)
    translations: list[str] = []
    lock: Lock = bundle["lock"]  # type: ignore[assignment]
    model.eval()
    with lock:
        with torch.no_grad():
            for batch_index, start in enumerate(range(0, len(cues), MARIAN_BATCH_SIZE), start=1):
                batch = cues[start : start + MARIAN_BATCH_SIZE]
                _notify(progress, "translate", f"Translating batch {batch_index}/{total_batches}.")
                encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                if device != "cpu":
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                generated = model.generate(**encoded, max_new_tokens=256)
                translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return translations


def _translate_ort_seq2seq_bundle(
    bundle: dict[str, object],
    cues: list[str],
    progress: ProgressCallback | None = None,
) -> list[str]:
    if not cues:
        return []
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    total_batches = max((len(cues) + MARIAN_BATCH_SIZE - 1) // MARIAN_BATCH_SIZE, 1)
    translations: list[str] = []
    lock: Lock = bundle["lock"]  # type: ignore[assignment]
    with lock:
        for batch_index, start in enumerate(range(0, len(cues), MARIAN_BATCH_SIZE), start=1):
            batch = cues[start : start + MARIAN_BATCH_SIZE]
            _notify(progress, "translate", f"Translating batch {batch_index}/{total_batches}.")
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            generated = model.generate(**encoded, max_new_tokens=256)
            translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return translations


def _translate_openvino_seq2seq_bundle(
    bundle: dict[str, object],
    cues: list[str],
    progress: ProgressCallback | None = None,
) -> list[str]:
    if not cues:
        return []
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    batch_size = OPENVINO_MARIAN_BATCH_SIZE
    total_batches = max((len(cues) + batch_size - 1) // batch_size, 1)
    translations: list[str] = []
    lock: Lock = bundle["lock"]  # type: ignore[assignment]
    with lock:
        for batch_index, start in enumerate(range(0, len(cues), batch_size), start=1):
            batch = cues[start : start + batch_size]
            _notify(progress, "translate", f"Translating batch {batch_index}/{total_batches}.")
            encoded = tokenizer(batch, return_tensors="np", padding=True, truncation=True)
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
        token = {"tur": "tur"}.get(target.canonical)
    elif "opus-mt-en-jap" in model_id:
        token = {"jpn": "jap"}.get(target.canonical)
    if not token:
        return cues
    return [f">>{token}<< {cue}" for cue in cues]


def _resolve_device_preference(device_preference: str, progress: ProgressCallback | None = None) -> str:
    normalized = _normalize_device_preference(device_preference)
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        runtime_reason = runtime_target_block_reason(
            normalized,
            has_cuda=_current_runtime_has_cuda(),
            has_directml=_current_runtime_has_directml(),
            has_openvino_gpu=_current_runtime_has_openvino_gpu(),
        )
        if runtime_reason:
            raise TranslationError(runtime_reason)
        return "cuda"
    if normalized == "directml":
        runtime_reason = runtime_target_block_reason(
            normalized,
            has_cuda=_current_runtime_has_cuda(),
            has_directml=_current_runtime_has_directml(),
            has_openvino_gpu=_current_runtime_has_openvino_gpu(),
        )
        if runtime_reason:
            raise TranslationError(runtime_reason)
        return "directml"
    if normalized == "openvino_gpu":
        runtime_reason = runtime_target_block_reason(
            normalized,
            has_cuda=_current_runtime_has_cuda(),
            has_directml=_current_runtime_has_directml(),
            has_openvino_gpu=_current_runtime_has_openvino_gpu(),
        )
        if runtime_reason:
            raise TranslationError(runtime_reason)
        return "openvino_gpu"
    if _current_runtime_has_cuda():
        return "cuda"
    if _current_runtime_has_openvino_gpu():
        return "openvino_gpu"
    if _current_runtime_has_directml():
        return "directml"
    if progress is not None:
        _notify(progress, "runtime", "Falling back to CPU execution.")
    return "cpu"


def _normalize_device_preference(device_preference: str | None) -> str:
    return (device_preference or "auto").strip().lower()


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
        if not _module_available(module_name)
    )
    return PythonRuntime(
        executable=sys.executable,
        missing_modules=missing,
        all_modules_present=not missing,
        current=True,
        has_cuda=_current_runtime_has_cuda(),
        python_version=_current_python_version(),
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
        "def module_available(name):\n"
        "    try:\n"
        "        return importlib.util.find_spec(name) is not None\n"
        "    except (ImportError, ModuleNotFoundError, ValueError):\n"
        "        return False\n"
        "mods=sys.argv[1:];"
        "version='.'.join(str(part) for part in sys.version_info[:3]);"
        "missing=[m for m in mods if not module_available(m)];"
        "torch_spec=importlib.util.find_spec('torch');"
        "has_cuda=bool(__import__('torch').cuda.is_available()) if torch_spec is not None else False;"
        "print(json.dumps({'missing':missing,'has_cuda':has_cuda,'python_version':version}))"
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
        python_version=str(payload.get("python_version") or ""),
    )


def _ensure_python_packages(packages: list[str], feature_name: str, progress: ProgressCallback | None = None) -> None:
    missing = [package for package in packages if not _package_runtime_ready(package)]
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
        raise TranslationError(f"Polylinguist could not launch pip for {feature_name}: {exc}") from exc


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


def _current_runtime_has_directml() -> bool:
    try:
        import onnxruntime as ort  # type: ignore

        providers = {provider.lower() for provider in ort.get_available_providers()}
        return "dmlexecutionprovider" in providers or "directmlexecutionprovider" in providers
    except Exception:
        return False


def _current_runtime_has_openvino_gpu() -> bool:
    try:
        from openvino.runtime import Core  # type: ignore

        core = Core()
        return any(device.upper().startswith("GPU") for device in core.available_devices)
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


def runtime_packages_for(provider: str, target: str) -> list[str]:
    normalized_provider = (provider or "").strip().lower()
    normalized_target = _normalize_device_preference(target)
    if normalized_provider == "argos":
        return ["argostranslate"]
    if normalized_provider == "marian":
        if normalized_target == "directml":
            return _ort_runtime_packages()
        if normalized_target == "openvino_gpu":
            return _openvino_runtime_packages()
        return _hf_runtime_packages()
    if normalized_provider == "nllb":
        return _hf_runtime_packages()
    return []


def _hf_runtime_packages() -> list[str]:
    return ["huggingface-hub", "transformers", "torch>=2.7", "sentencepiece"]


def _ort_runtime_packages() -> list[str]:
    return ["transformers", "sentencepiece", "torch>=2.7,<2.9", "optimum", "onnx", "onnxruntime-directml"]


def _openvino_runtime_packages() -> list[str]:
    return [
        "transformers>=4.53,<4.54",
        "sentencepiece>=0.2.0",
        "torch>=2.7,<2.9",
        "pillow>=10.0.0",
        "optimum-intel[openvino]>=1.25.1,<1.26",
    ]


def _module_name_for_package(package: str) -> str:
    package_name = _package_name(package)
    return PACKAGE_MODULES.get(package_name, package_name.replace("-", "_"))


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _package_name(package: str) -> str:
    return Requirement(package).name


def _package_runtime_ready(package: str) -> bool:
    package_name = _package_name(package)
    if not _module_available(_module_name_for_package(package_name)):
        return False
    try:
        installed_version = importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return False
    requirement = Requirement(package)
    if not requirement.specifier:
        return True
    return installed_version in requirement.specifier


def _runtime_metadata_snapshot(runtime: PythonRuntime, packages: list[str]) -> dict[str, object]:
    inspected = _inspect_runtime_environment(runtime.executable, packages)
    return {
        "python_executable": runtime.executable,
        "python_version": inspected.get("python_version") or runtime.python_version,
        "has_cuda": bool(inspected.get("has_cuda", runtime.has_cuda)),
        "has_directml": bool(inspected.get("has_directml")),
        "has_openvino_gpu": bool(inspected.get("has_openvino_gpu")),
        "requirements": list(packages),
        "package_versions": inspected.get("packages", {}),
        "captured_at": _utc_now_iso(),
    }


def _inspect_runtime_environment(executable: str, packages: list[str]) -> dict[str, object]:
    package_names = list(dict.fromkeys(_package_name(package) for package in packages))
    if _same_executable(executable, sys.executable):
        versions = {package_name: _installed_package_version(package_name) for package_name in package_names}
        return {
            "python_version": _current_python_version(),
            "has_cuda": _current_runtime_has_cuda(),
            "has_directml": _current_runtime_has_directml(),
            "has_openvino_gpu": _current_runtime_has_openvino_gpu(),
            "packages": versions,
        }
    probe_code = (
        "import importlib.util,json,sys;"
        "from importlib import metadata as m;"
        "def version(name):\n"
        "    try:\n"
        "        return m.version(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        return None\n"
        "mods=sys.argv[1:];"
        "version_str='.'.join(str(part) for part in sys.version_info[:3]);"
        "torch_spec=importlib.util.find_spec('torch');"
        "has_cuda=bool(__import__('torch').cuda.is_available()) if torch_spec is not None else False;"
        "ort_spec=importlib.util.find_spec('onnxruntime');"
        "has_directml=False;"
        "if ort_spec is not None:\n"
        "    try:\n"
        "        providers={provider.lower() for provider in __import__('onnxruntime').get_available_providers()};\n"
        "        has_directml='dmlexecutionprovider' in providers or 'directmlexecutionprovider' in providers\n"
        "    except Exception:\n"
        "        has_directml=False\n"
        "ov_spec=importlib.util.find_spec('openvino.runtime');"
        "has_openvino=False;"
        "if ov_spec is not None:\n"
        "    try:\n"
        "        Core=__import__('openvino.runtime', fromlist=['Core']).Core;\n"
        "        core=Core();\n"
        "        has_openvino=any(device.upper().startswith('GPU') for device in core.available_devices)\n"
        "    except Exception:\n"
        "        has_openvino=False\n"
        "packages={name:version(name) for name in mods};"
        "print(json.dumps({'python_version':version_str,'has_cuda':has_cuda,'has_directml':has_directml,'has_openvino_gpu':has_openvino,'packages':packages}))"
    )
    try:
        completed = subprocess.run(
            [executable, "-c", probe_code, *package_names],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        payload = json.loads(completed.stdout.strip() or "{}")
    except Exception:
        payload = {}
    return {
        "python_version": payload.get("python_version"),
        "has_cuda": bool(payload.get("has_cuda")),
        "has_directml": bool(payload.get("has_directml")),
        "has_openvino_gpu": bool(payload.get("has_openvino_gpu")),
        "packages": payload.get("packages", {}),
    }


def _installed_package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _current_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _ensure_openvino_python_compatibility() -> None:
    reason = system_target_block_reason("openvino_gpu")
    if reason:
        raise TranslationError(reason)


def _load_openvino_seq2seq_class():
    try:
        from optimum.intel import OVModelForSeq2SeqLM  # type: ignore

        return OVModelForSeq2SeqLM
    except Exception:
        from optimum.intel.openvino import OVModelForSeq2SeqLM  # type: ignore

        return OVModelForSeq2SeqLM


def _notify(progress: ProgressCallback | None, stage: str, message: str) -> None:
    if progress is not None:
        progress(stage, message)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
