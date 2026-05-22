from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass

from polylinguist.schemas import SystemProfileResponse


@dataclass(frozen=True)
class SystemProfile:
    os: str
    arch: str
    cpu_cores: int
    total_ram_gb: float
    free_disk_gb: float
    has_cuda: bool
    has_mps: bool

    @property
    def tier(self) -> str:
        if self.total_ram_gb < 8 or self.cpu_cores < 4 or self.free_disk_gb < 8:
            return "low"
        if self.total_ram_gb >= 16 and (self.has_cuda or self.has_mps or self.cpu_cores >= 8):
            return "strong"
        return "standard"

    def to_response(self) -> SystemProfileResponse:
        return SystemProfileResponse(
            os=self.os,
            arch=self.arch,
            cpu_cores=self.cpu_cores,
            total_ram_gb=round(self.total_ram_gb, 2),
            free_disk_gb=round(self.free_disk_gb, 2),
            has_cuda=self.has_cuda,
            has_mps=self.has_mps,
            tier=self.tier,
        )


class SystemProfileService:
    def detect(self) -> SystemProfile:
        return SystemProfile(
            os=platform.system().lower(),
            arch=platform.machine().lower(),
            cpu_cores=os.cpu_count() or 1,
            total_ram_gb=_detect_total_ram_gb(),
            free_disk_gb=_detect_free_disk_gb(),
            has_cuda=_detect_cuda(),
            has_mps=_detect_mps(),
        )


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
    usage = shutil.disk_usage(os.getenv("POLYLINGUIST_HOME", str(os.path.expanduser("~"))))
    return usage.free / (1024 ** 3)


def _detect_cuda() -> bool:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return True
    except Exception:
        pass

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(completed.stdout.strip())
    except Exception:
        return False


def _detect_mps() -> bool:
    try:
        import torch  # type: ignore

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False
