from __future__ import annotations

import os
from pathlib import Path


def huggingface_hub_cache_dir() -> Path:
    explicit_cache = os.getenv("HF_HUB_CACHE")
    if explicit_cache:
        return Path(explicit_cache)
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_model_cache_exists(model_id: str) -> bool:
    owner, repo = model_id.split("/", 1)
    cache_dir = huggingface_hub_cache_dir() / f"models--{owner}--{repo}"
    snapshots_dir = cache_dir / "snapshots"
    refs_dir = cache_dir / "refs"
    return (snapshots_dir.exists() and any(snapshots_dir.iterdir())) or (refs_dir.exists() and any(refs_dir.iterdir()))


def local_model_artifact_dir(root: Path, provider: str, model_id: str, target: str) -> Path:
    return root / provider / target / _slugify_model_id(model_id)


def local_model_artifact_exists(root: Path, provider: str, model_id: str, target: str) -> bool:
    artifact_dir = local_model_artifact_dir(root, provider, model_id, target)
    return artifact_dir.exists() and any(artifact_dir.iterdir())


def _slugify_model_id(model_id: str) -> str:
    return (
        model_id.replace("\\", "__")
        .replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )
