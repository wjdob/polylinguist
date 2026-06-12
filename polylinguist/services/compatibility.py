from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polylinguist.services.system_profile import SystemProfile


TARGET_LABELS = {
    "auto": "Auto",
    "cpu": "CPU",
    "cuda": "CUDA",
    "directml": "DirectML",
    "openvino_gpu": "OpenVINO GPU",
}


def normalize_target(value: str | None) -> str:
    return (value or "auto").strip().lower()


def target_label(target: str | None) -> str:
    normalized = normalize_target(target)
    return TARGET_LABELS.get(normalized, normalized or "Auto")


def provider_target_block_reason(provider: str, target: str | None) -> str | None:
    normalized_provider = (provider or "").strip().lower()
    normalized_target = normalize_target(target)
    if normalized_target in {"auto", "cpu"}:
        return None
    if normalized_provider == "argos":
        return "Argos Translate is CPU-only in this release."
    if normalized_provider == "nllb" and normalized_target in {"directml", "openvino_gpu"}:
        return "NLLB is only supported on CPU or CUDA in this release."
    return None


def system_target_block_reason(
    target: str | None,
    *,
    system_name: str | None = None,
) -> str | None:
    normalized_target = normalize_target(target)
    system_name = (system_name or platform.system()).lower()
    _ = normalized_target
    _ = system_name
    return None


def python_target_block_reason(
    target: str | None,
    *,
    system_name: str | None = None,
    python_version: object | None = None,
) -> str | None:
    normalized_target = normalize_target(target)
    system_name = (system_name or platform.system()).lower()
    version = _version_tuple(python_version or sys.version_info)
    if normalized_target == "openvino_gpu" and system_name == "windows" and version >= (3, 14):
        return (
            "OpenVINO GPU on Windows currently requires Python 3.13 or earlier. "
            f"Detected Python {version[0]}.{version[1]}.{version[2]}."
        )
    return None


def machine_target_block_reason(profile: "SystemProfile", target: str | None) -> str | None:
    normalized_target = normalize_target(target)
    if normalized_target in {"auto", "cpu"}:
        return None
    if normalized_target == "cuda" and not profile.has_cuda:
        return "The current machine does not expose CUDA."
    if normalized_target == "directml" and not _profile_has_target(profile, "directml"):
        return "The current machine does not expose DirectML."
    if normalized_target == "openvino_gpu":
        if not _profile_has_target(profile, "openvino_gpu"):
            return "The current machine does not expose OpenVINO GPU."
    return None


def runtime_target_block_reason(
    target: str | None,
    *,
    has_cuda: bool,
    has_directml: bool,
    has_openvino_gpu: bool,
    system_name: str | None = None,
    python_version: object | None = None,
) -> str | None:
    normalized_target = normalize_target(target)
    system_reason = system_target_block_reason(
        normalized_target,
        system_name=system_name,
    )
    if system_reason:
        return system_reason
    python_reason = python_target_block_reason(
        normalized_target,
        system_name=system_name,
        python_version=python_version,
    )
    if python_reason:
        return python_reason
    if normalized_target == "cuda" and not has_cuda:
        return "GPU processing was requested, but CUDA is not available in the selected Python runtime."
    if normalized_target == "directml" and not has_directml:
        return "GPU processing was requested, but DirectML is not available in the selected Python runtime."
    if normalized_target == "openvino_gpu" and not has_openvino_gpu:
        return "GPU processing was requested, but OpenVINO GPU is not available in the selected Python runtime."
    return None


def _profile_has_target(profile: "SystemProfile", target: str) -> bool:
    return any(target in accelerator.supported_targets for accelerator in profile.accelerators)


def _version_tuple(value: object) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        items = list(value) + [0, 0, 0]
        return int(items[0]), int(items[1]), int(items[2])
    major = int(getattr(value, "major", 0))
    minor = int(getattr(value, "minor", 0))
    micro = int(getattr(value, "micro", 0))
    return major, minor, micro
