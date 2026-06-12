from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from packaging.requirements import Requirement

from polylinguist.services.compatibility import normalize_target, python_target_block_reason


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


@dataclass(frozen=True)
class PythonRuntime:
    executable: str
    missing_modules: tuple[str, ...]
    all_modules_present: bool
    current: bool
    has_cuda: bool = False
    has_directml: bool = False
    has_openvino_gpu: bool = False
    python_version: str | None = None


def same_executable(left: str, right: str) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def current_runtime_has_cuda() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def current_runtime_has_directml() -> bool:
    try:
        import onnxruntime as ort  # type: ignore

        providers = {provider.lower() for provider in ort.get_available_providers()}
        return "dmlexecutionprovider" in providers or "directmlexecutionprovider" in providers
    except Exception:
        return False


def current_runtime_has_openvino_gpu() -> bool:
    try:
        from openvino.runtime import Core  # type: ignore

        core = Core()
        return any(device.upper().startswith("GPU") for device in core.available_devices)
    except Exception:
        return False


def current_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def module_name_for_package(package: str) -> str:
    package_name = package_name_for_requirement(package)
    return PACKAGE_MODULES.get(package_name, package_name.replace("-", "_"))


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def package_name_for_requirement(package: str) -> str:
    return Requirement(package).name


def package_runtime_ready(package: str) -> bool:
    package_name = package_name_for_requirement(package)
    if not module_available(module_name_for_package(package_name)):
        return False
    try:
        installed_version = importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return False
    requirement = Requirement(package)
    if not requirement.specifier:
        return True
    return installed_version in requirement.specifier


def installed_package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def runtime_packages_for(provider: str, target: str) -> list[str]:
    normalized_provider = (provider or "").strip().lower()
    normalized_target = normalize_target(target)
    if normalized_provider == "argos":
        return ["argostranslate"]
    if normalized_provider == "marian":
        if normalized_target == "directml":
            return ort_runtime_packages()
        if normalized_target == "openvino_gpu":
            return openvino_runtime_packages()
        return hf_runtime_packages()
    if normalized_provider == "nllb":
        return hf_runtime_packages()
    return []


def hf_runtime_packages() -> list[str]:
    return ["huggingface-hub", "transformers", "torch>=2.7", "sentencepiece"]


def ort_runtime_packages() -> list[str]:
    return ["transformers", "sentencepiece", "torch>=2.7,<2.9", "optimum", "onnx", "onnxruntime-directml"]


def openvino_runtime_packages() -> list[str]:
    return [
        "transformers>=4.53,<4.54",
        "sentencepiece>=0.2.0",
        "torch>=2.7,<2.9",
        "pillow>=10.0.0",
        "optimum-intel[openvino]>=1.25.1,<1.26",
    ]


def resolve_runtime(packages: list[str], prefer_cuda: bool = False) -> PythonRuntime:
    required_modules = tuple(module_name_for_package(package) for package in packages)
    best_match: PythonRuntime | None = None
    for executable in discover_python_executables():
        runtime = probe_python_runtime(executable, required_modules)
        if not runtime or not runtime.all_modules_present:
            continue
        if prefer_cuda and runtime.has_cuda:
            return runtime
        if best_match is None:
            best_match = runtime
    current = probe_current_runtime(required_modules)
    if prefer_cuda and current.has_cuda and current.all_modules_present:
        return current
    if best_match is not None:
        return best_match
    return current


def resolve_runtime_for_target(
    target: str,
    packages: list[str],
    prefer_cuda: bool = False,
    system_name: str | None = None,
) -> PythonRuntime:
    compatible = find_compatible_runtime_for_target(target, system_name=system_name)
    if compatible is not None:
        return compatible
    return resolve_runtime(packages, prefer_cuda=prefer_cuda)


def find_compatible_runtime_for_target(target: str, system_name: str | None = None) -> PythonRuntime | None:
    normalized_target = normalize_target(target)
    required_modules: tuple[str, ...] = ()
    for executable in discover_python_executables():
        runtime = probe_python_runtime(executable, required_modules)
        if runtime and runtime_is_compatible_for_target(runtime, normalized_target, system_name=system_name):
            return runtime
    return None


def compatible_runtime_block_reason(target: str, system_name: str | None = None) -> str | None:
    normalized_target = normalize_target(target)
    if normalized_target in {"auto", "cpu", "directml"}:
        return None
    if find_compatible_runtime_for_target(normalized_target, system_name=system_name) is not None:
        return None
    if normalized_target == "cuda":
        return "Polylinguist did not discover a CUDA-capable Python runtime for this target."
    if normalized_target == "openvino_gpu":
        return (
            "OpenVINO GPU on Windows requires a discovered Python 3.13 or earlier runtime. "
            "Polylinguist did not find one."
        )
    return None


def runtime_is_compatible_for_target(runtime: PythonRuntime, target: str, system_name: str | None = None) -> bool:
    normalized_target = normalize_target(target)
    if normalized_target in {"auto", "cpu", "directml"}:
        return True
    python_reason = python_target_block_reason(
        normalized_target,
        system_name=system_name or platform.system(),
        python_version=_version_tuple_from_text(runtime.python_version),
    )
    if python_reason:
        return False
    if normalized_target == "cuda":
        return runtime.has_cuda
    return True


def probe_current_runtime(required_modules: tuple[str, ...]) -> PythonRuntime:
    missing = tuple(
        module_name
        for module_name in required_modules
        if not module_available(module_name)
    )
    return PythonRuntime(
        executable=sys.executable,
        missing_modules=missing,
        all_modules_present=not missing,
        current=True,
        has_cuda=current_runtime_has_cuda(),
        has_directml=current_runtime_has_directml(),
        has_openvino_gpu=current_runtime_has_openvino_gpu(),
        python_version=current_python_version(),
    )


def discover_python_executables() -> list[str]:
    candidates: list[str] = [sys.executable]
    env_override = os.getenv("POLYLINGUIST_PYTHON")
    if env_override:
        candidates.append(env_override)
    candidates.extend(discover_local_virtualenvs())
    candidates.extend(discover_with_command(["py", "-0p"]))
    candidates.extend(discover_with_command(["where", "python"]))
    candidates.extend(discover_with_command(["where", "py"]))

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


def discover_local_virtualenvs() -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    candidates: list[str] = []
    for name in [".venv", ".benchmarks\\venv", "venv"]:
        executable = repo_root / name / "Scripts" / "python.exe"
        if executable.exists():
            candidates.append(str(executable))
    return candidates


def discover_with_command(command: list[str]) -> list[str]:
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


def probe_python_runtime(executable: str, required_modules: tuple[str, ...]) -> PythonRuntime | None:
    if same_executable(executable, sys.executable):
        return probe_current_runtime(required_modules)
    probe_code = (
        "import importlib.util, json, sys\n"
        "def module_available(name):\n"
        "    try:\n"
        "        return importlib.util.find_spec(name) is not None\n"
        "    except (ImportError, ModuleNotFoundError, ValueError):\n"
        "        return False\n"
        "mods = sys.argv[1:]\n"
        "version = '.'.join(str(part) for part in sys.version_info[:3])\n"
        "missing = [name for name in mods if not module_available(name)]\n"
        "torch_spec = importlib.util.find_spec('torch')\n"
        "has_cuda = bool(__import__('torch').cuda.is_available()) if torch_spec is not None else False\n"
        "ort_spec = importlib.util.find_spec('onnxruntime')\n"
        "has_directml = False\n"
        "if ort_spec is not None:\n"
        "    try:\n"
        "        providers = {provider.lower() for provider in __import__('onnxruntime').get_available_providers()}\n"
        "        has_directml = 'dmlexecutionprovider' in providers or 'directmlexecutionprovider' in providers\n"
        "    except Exception:\n"
        "        has_directml = False\n"
        "try:\n"
        "    ov_spec = importlib.util.find_spec('openvino.runtime')\n"
        "except (ImportError, ModuleNotFoundError, ValueError):\n"
        "    ov_spec = None\n"
        "has_openvino = False\n"
        "if ov_spec is not None:\n"
        "    try:\n"
        "        Core = __import__('openvino.runtime', fromlist=['Core']).Core\n"
        "        core = Core()\n"
        "        has_openvino = any(device.upper().startswith('GPU') for device in core.available_devices)\n"
        "    except Exception:\n"
        "        has_openvino = False\n"
        "print(json.dumps({'missing': missing, 'has_cuda': has_cuda, 'has_directml': has_directml, 'has_openvino_gpu': has_openvino, 'python_version': version}))\n"
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
        has_directml=bool(payload.get("has_directml")),
        has_openvino_gpu=bool(payload.get("has_openvino_gpu")),
        python_version=str(payload.get("python_version") or ""),
    )


def inspect_runtime_environment(executable: str, packages: list[str]) -> dict[str, object]:
    package_names = list(dict.fromkeys(package_name_for_requirement(package) for package in packages))
    if same_executable(executable, sys.executable):
        versions = {package_name: installed_package_version(package_name) for package_name in package_names}
        return {
            "python_version": current_python_version(),
            "has_cuda": current_runtime_has_cuda(),
            "has_directml": current_runtime_has_directml(),
            "has_openvino_gpu": current_runtime_has_openvino_gpu(),
            "packages": versions,
        }
    probe_code = (
        "import importlib.util, json, sys\n"
        "from importlib import metadata as m\n"
        "def version(name):\n"
        "    try:\n"
        "        return m.version(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        return None\n"
        "mods = sys.argv[1:]\n"
        "version_str = '.'.join(str(part) for part in sys.version_info[:3])\n"
        "torch_spec = importlib.util.find_spec('torch')\n"
        "has_cuda = bool(__import__('torch').cuda.is_available()) if torch_spec is not None else False\n"
        "ort_spec = importlib.util.find_spec('onnxruntime')\n"
        "has_directml = False\n"
        "if ort_spec is not None:\n"
        "    try:\n"
        "        providers = {provider.lower() for provider in __import__('onnxruntime').get_available_providers()}\n"
        "        has_directml = 'dmlexecutionprovider' in providers or 'directmlexecutionprovider' in providers\n"
        "    except Exception:\n"
        "        has_directml = False\n"
        "try:\n"
        "    ov_spec = importlib.util.find_spec('openvino.runtime')\n"
        "except (ImportError, ModuleNotFoundError, ValueError):\n"
        "    ov_spec = None\n"
        "has_openvino = False\n"
        "if ov_spec is not None:\n"
        "    try:\n"
        "        Core = __import__('openvino.runtime', fromlist=['Core']).Core\n"
        "        core = Core()\n"
        "        has_openvino = any(device.upper().startswith('GPU') for device in core.available_devices)\n"
        "    except Exception:\n"
        "        has_openvino = False\n"
        "packages = {name: version(name) for name in mods}\n"
        "print(json.dumps({'python_version': version_str, 'has_cuda': has_cuda, 'has_directml': has_directml, 'has_openvino_gpu': has_openvino, 'packages': packages}))\n"
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


def runtime_metadata_snapshot(runtime: PythonRuntime, packages: list[str]) -> dict[str, object]:
    inspected = inspect_runtime_environment(runtime.executable, packages)
    return {
        "python_executable": runtime.executable,
        "python_version": inspected.get("python_version") or runtime.python_version,
        "has_cuda": bool(inspected.get("has_cuda", runtime.has_cuda)),
        "has_directml": bool(inspected.get("has_directml", runtime.has_directml)),
        "has_openvino_gpu": bool(inspected.get("has_openvino_gpu", runtime.has_openvino_gpu)),
        "requirements": list(packages),
        "package_versions": inspected.get("packages", {}),
        "captured_at": utc_now_iso(),
    }


def python_runtime_has_cuda(executable: str) -> bool:
    if same_executable(executable, sys.executable):
        return current_runtime_has_cuda()
    runtime = probe_python_runtime(executable, ())
    return bool(runtime and runtime.has_cuda)


def _version_tuple_from_text(value: str | None) -> tuple[int, int, int]:
    if not value:
        return (0, 0, 0)
    parts = [part for part in str(value).split(".") if part.strip()]
    normalized = parts + ["0", "0", "0"]
    try:
        return int(normalized[0]), int(normalized[1]), int(normalized[2])
    except ValueError:
        return (0, 0, 0)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
