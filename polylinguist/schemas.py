from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ProcessingDevice = Literal["auto", "cpu", "cuda", "directml", "openvino_gpu"]


class AddonSettings(BaseModel):
    source_lang: str = "eng"
    target_lang: str = "spa"
    preferred_provider: str = "auto"
    selected_model_id: str | None = None
    processing_device: ProcessingDevice = "auto"
    format_mode: Literal["dual", "translated_only"] = "dual"


class AcceleratorResponse(BaseModel):
    vendor: str
    name: str
    supported_targets: list[str] = Field(default_factory=list)


class SettingsEnvelope(BaseModel):
    settings: AddonSettings
    updated_at: datetime | None = None


class SystemProfileResponse(BaseModel):
    os: str
    arch: str
    cpu_cores: int
    total_ram_gb: float
    free_disk_gb: float
    has_cuda: bool
    has_mps: bool
    accelerators: list[AcceleratorResponse] = Field(default_factory=list)
    tier: Literal["low", "standard", "strong"]


class ModelOptionResponse(BaseModel):
    provider: str
    model_id: str
    label: str
    source_lang: str
    target_lang: str
    size_mb: int
    available: bool
    direct: bool
    installed: bool
    installed_targets: list[str] = Field(default_factory=list)
    supported_targets: list[str] = Field(default_factory=list)
    recommended_target: str | None = None
    availability_reason: str | None = None
    note: str | None = None
    license: str | None = None
    recommended: bool = False
    install_strategy: str = "direct"


class ModelCatalogResponse(BaseModel):
    source_lang: str
    target_lang: str
    recommended_provider: str | None = None
    recommended_model_id: str | None = None
    profile: SystemProfileResponse
    models: list[ModelOptionResponse]


class ModelInstallRequest(BaseModel):
    provider: str
    model_id: str
    source_lang: str
    target_lang: str
    processing_device: ProcessingDevice = "auto"
    persist_selection: bool = True


class ModelInstallResponse(BaseModel):
    provider: str
    model_id: str
    installed: bool
    detail: str


class InstallJobResponse(BaseModel):
    job_id: str
    provider: str
    model_id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: str
    message: str
    detail: str | None = None
    log_lines: list[str] = Field(default_factory=list)


class SubtitleQuery(BaseModel):
    type: str
    media_id: str = Field(alias="id")
    video_hash: str | None = None
    video_size: str | None = None
    filename: str | None = None


class LanguageResponse(BaseModel):
    code: str
    label: str
    iso639_1: str | None = None
    nllb: str | None = None
