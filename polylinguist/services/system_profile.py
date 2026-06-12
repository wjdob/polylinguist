from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Iterable

from polylinguist.schemas import AcceleratorResponse, SystemProfileResponse
from polylinguist.services.compatibility import system_target_block_reason


@dataclass(frozen=True)
class AcceleratorInfo:
    vendor: str
    name: str
    supported_targets: tuple[str, ...] = ()

    def to_response(self) -> AcceleratorResponse:
        return AcceleratorResponse(
            vendor=self.vendor,
            name=self.name,
            supported_targets=list(self.supported_targets),
        )


@dataclass(frozen=True)
class SystemProfile:
    os: str
    arch: str
    cpu_cores: int
    total_ram_gb: float
    free_disk_gb: float
    has_cuda: bool
    has_mps: bool
    accelerators: tuple[AcceleratorInfo, ...] = field(default_factory=tuple)

    @property
    def tier(self) -> str:
        if self.total_ram_gb < 8 or self.cpu_cores < 4 or self.free_disk_gb < 8:
            return "low"
        if self.total_ram_gb >= 16 and (self.has_cuda or self.has_mps or self.has_any_acceleration or self.cpu_cores >= 8):
            return "strong"
        return "standard"

    @property
    def has_any_acceleration(self) -> bool:
        return any(accelerator.supported_targets for accelerator in self.accelerators)

    def supports_target(self, target: str) -> bool:
        normalized = (target or "").lower()
        if normalized == "cpu":
            return True
        if normalized == "cuda":
            return self.has_cuda
        if normalized == "mps":
            return self.has_mps
        return any(normalized in accelerator.supported_targets for accelerator in self.accelerators)

    def accelerators_for_target(self, target: str) -> list[AcceleratorInfo]:
        normalized = (target or "").lower()
        return [item for item in self.accelerators if normalized in item.supported_targets]

    def to_response(self) -> SystemProfileResponse:
        return SystemProfileResponse(
            os=self.os,
            arch=self.arch,
            cpu_cores=self.cpu_cores,
            total_ram_gb=round(self.total_ram_gb, 2),
            free_disk_gb=round(self.free_disk_gb, 2),
            has_cuda=self.has_cuda,
            has_mps=self.has_mps,
            accelerators=[item.to_response() for item in self.accelerators],
            tier=self.tier,
        )


class SystemProfileService:
    def detect(self) -> SystemProfile:
        system = platform.system().lower()
        accelerators = tuple(_detect_accelerators(system))
        has_cuda = any("cuda" in accelerator.supported_targets for accelerator in accelerators) or _detect_cuda()
        has_mps = _detect_mps()
        return SystemProfile(
            os=system,
            arch=platform.machine().lower(),
            cpu_cores=os.cpu_count() or 1,
            total_ram_gb=_detect_total_ram_gb(),
            free_disk_gb=_detect_free_disk_gb(),
            has_cuda=has_cuda,
            has_mps=has_mps,
            accelerators=accelerators,
        )


def _detect_accelerators(system: str) -> Iterable[AcceleratorInfo]:
    accelerators: list[AcceleratorInfo] = []

    nvidia_names = _detect_nvidia_gpu_names()
    for name in nvidia_names:
        accelerators.append(AcceleratorInfo(vendor="nvidia", name=name, supported_targets=("cuda",)))

    if system == "windows":
        adapters = _detect_windows_video_adapters()
        for adapter in adapters:
            supported_targets: list[str] = []
            if adapter.vendor == "amd" and _supports_directml_adapter(adapter):
                supported_targets.append("directml")
            if adapter.vendor == "intel" and _supports_openvino_gpu_adapter(adapter):
                supported_targets.append("openvino_gpu")
            if supported_targets:
                accelerators.append(
                    AcceleratorInfo(
                        vendor=adapter.vendor,
                        name=adapter.name,
                        supported_targets=tuple(supported_targets),
                    )
                )
    return _dedupe_accelerators(accelerators)


@dataclass(frozen=True)
class _VideoAdapter:
    vendor: str
    name: str


def _dedupe_accelerators(items: Iterable[AcceleratorInfo]) -> tuple[AcceleratorInfo, ...]:
    deduped: dict[tuple[str, str], AcceleratorInfo] = {}
    for item in items:
        key = (item.vendor, item.name)
        if key in deduped:
            merged_targets = tuple(sorted(set(deduped[key].supported_targets) | set(item.supported_targets)))
            deduped[key] = AcceleratorInfo(vendor=item.vendor, name=item.name, supported_targets=merged_targets)
        else:
            deduped[key] = item
    return tuple(deduped.values())


def _detect_total_ram_gb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
                return (pages * page_size) / (1024 ** 3)
        except Exception:
            pass

    if platform.system().lower() == "windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / (1024 ** 3)

    return 8.0


def _detect_free_disk_gb() -> float:
    candidates = [
        os.getenv("POLYLINGUIST_HOME"),
        str(os.path.expanduser("~")),
        os.getcwd(),
        os.path.abspath(os.sep),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            usage = shutil.disk_usage(candidate)
            return usage.free / (1024 ** 3)
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            continue
    return 20.0


def _detect_cuda() -> bool:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    return bool(_detect_nvidia_gpu_names())


def _detect_nvidia_gpu_names() -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _detect_mps() -> bool:
    try:
        import torch  # type: ignore

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def _detect_windows_video_adapters() -> list[_VideoAdapter]:
    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterCompatibility | ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        payload = json.loads(completed.stdout.strip() or "[]")
    except Exception:
        return []

    if isinstance(payload, dict):
        payload = [payload]
    adapters: list[_VideoAdapter] = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "").strip()
        compat = str(item.get("AdapterCompatibility") or "").strip()
        vendor = _normalize_vendor(" ".join([compat, name]))
        if name and vendor:
            adapters.append(_VideoAdapter(vendor=vendor, name=name))
    return adapters


def _normalize_vendor(value: str) -> str | None:
    lowered = value.lower()
    if "nvidia" in lowered:
        return "nvidia"
    if "advanced micro devices" in lowered or " amd" in lowered or lowered.startswith("amd") or " ati" in lowered:
        return "amd"
    if "intel" in lowered:
        return "intel"
    if "apple" in lowered:
        return "apple"
    return None


def _supports_directml_adapter(adapter: _VideoAdapter) -> bool:
    if adapter.vendor != "amd":
        return False
    lowered = adapter.name.lower()
    unsupported_markers = ("basic display", "remote display", "virtual", "hyper-v")
    return not any(marker in lowered for marker in unsupported_markers)


def _supports_openvino_gpu_adapter(adapter: _VideoAdapter) -> bool:
    if adapter.vendor != "intel":
        return False
    if system_target_block_reason("openvino_gpu") is not None:
        return False
    lowered = adapter.name.lower()
    unsupported_markers = ("basic display", "remote display", "virtual", "hyper-v")
    if any(marker in lowered for marker in unsupported_markers):
        return False
    arc_markers = ("intel arc", "arc(tm)", " arc a", " arc b", " arc pro")
    return any(marker in lowered for marker in arc_markers)


def _python_supports_openvino_gpu() -> bool:
    return system_target_block_reason("openvino_gpu") is None


def _directml_runtime_ready() -> bool:
    if platform.system().lower() != "windows":
        return False
    try:
        import onnxruntime as ort  # type: ignore

        providers = {provider.lower() for provider in ort.get_available_providers()}
        return "dmlexecutionprovider" in providers or "directmlexecutionprovider" in providers
    except Exception:
        return False


def _openvino_gpu_ready() -> bool:
    if platform.system().lower() != "windows":
        return False
    try:
        from openvino.runtime import Core  # type: ignore

        core = Core()
        return any(device.upper().startswith("GPU") for device in core.available_devices)
    except Exception:
        return False
