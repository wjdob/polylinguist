from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx

from polylinguist.schemas import ModelCatalogResponse, ModelOptionResponse
from polylinguist.services.languages import get_language, normalize_language
from polylinguist.services.local_models import hf_model_cache_exists, local_model_artifact_exists
from polylinguist.services.model_registry import InstalledModelRegistry
from polylinguist.services.system_profile import SystemProfile


ARGOS_INDEX_URL = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_SIZE_MB = 2480
MARIAN_MULTILINGUAL_TARGETS = {
    "spa",
    "fre",
    "ger",
    "ita",
    "por",
    "pol",
    "cze",
    "dut",
}
MARIAN_ENGLISH_FALLBACKS: dict[str, tuple[str, str]] = {
    "fre": ("Helsinki-NLP/opus-mt-en-ROMANCE", "Uses the multilingual English-to-Romance Marian model for this target language."),
    "spa": ("Helsinki-NLP/opus-mt-en-ROMANCE", "Uses the multilingual English-to-Romance Marian model for this target language."),
    "ita": ("Helsinki-NLP/opus-mt-en-ROMANCE", "Uses the multilingual English-to-Romance Marian model for this target language."),
    "por": ("Helsinki-NLP/opus-mt-en-ROMANCE", "Uses the multilingual English-to-Romance Marian model for this target language."),
    "pob": ("Helsinki-NLP/opus-mt-en-ROMANCE", "Uses the multilingual English-to-Romance Marian model for this target language."),
    "pol": ("Helsinki-NLP/opus-mt-en-ine", "Uses the multilingual English-to-Indo-European Marian model for this target language."),
    "ger": ("Helsinki-NLP/opus-mt-en-ine", "Uses the multilingual English-to-Indo-European Marian model for this target language."),
    "dut": ("Helsinki-NLP/opus-mt-en-ine", "Uses the multilingual English-to-Indo-European Marian model for this target language."),
    "swe": ("Helsinki-NLP/opus-mt-en-ine", "Uses the multilingual English-to-Indo-European Marian model for this target language."),
    "cze": ("Helsinki-NLP/opus-mt-en-ine", "Uses the multilingual English-to-Indo-European Marian model for this target language."),
    "rus": ("Helsinki-NLP/opus-mt-en-ine", "Uses the multilingual English-to-Indo-European Marian model for this target language."),
    "ukr": ("Helsinki-NLP/opus-mt-en-ine", "Uses the multilingual English-to-Indo-European Marian model for this target language."),
    "hin": ("Helsinki-NLP/opus-mt-en-ine", "Uses the multilingual English-to-Indo-European Marian model for this target language."),
    "tur": ("Helsinki-NLP/opus-mt-en-trk", "Uses the multilingual English-to-Turkic Marian model for this target language."),
    "jpn": ("Helsinki-NLP/opus-mt-en-jap", "Uses the English-to-Japanese Marian model for this target language."),
}


@dataclass(frozen=True)
class ModelDescriptor:
    provider: str
    model_id: str
    label: str
    source_lang: str
    target_lang: str
    size_mb: int
    available: bool
    direct: bool
    installed: bool
    installed_targets: tuple[str, ...]
    supported_targets: tuple[str, ...]
    recommended_target: str | None = None
    availability_reason: str | None = None
    note: str | None = None
    license: str | None = None
    install_strategy: str = "direct"


class ModelCatalogService:
    def __init__(self, metadata_cache_file: Path, registry: InstalledModelRegistry, model_artifacts_dir: Path) -> None:
        self.metadata_cache_file = metadata_cache_file
        self.registry = registry
        self.model_artifacts_dir = model_artifacts_dir

    def list_models(self, source_lang: str, target_lang: str, profile: SystemProfile) -> ModelCatalogResponse:
        source = normalize_language(source_lang) or source_lang
        target = normalize_language(target_lang) or target_lang

        descriptors: list[ModelDescriptor] = []
        descriptors.extend(self._argos_descriptors(source, target))
        descriptors.extend(self._marian_descriptors(source, target, profile))
        descriptors.extend(self._nllb_descriptors(source, target, profile))

        recommended = self.recommend(source, target, profile, descriptors)
        models = [
            ModelOptionResponse(
                provider=item.provider,
                model_id=item.model_id,
                label=item.label,
                source_lang=item.source_lang,
                target_lang=item.target_lang,
                size_mb=item.size_mb,
                available=item.available,
                direct=item.direct,
                installed=item.installed,
                installed_targets=list(item.installed_targets),
                supported_targets=list(item.supported_targets),
                recommended_target=item.recommended_target,
                availability_reason=item.availability_reason,
                note=item.note,
                license=item.license,
                recommended=bool(recommended and recommended.provider == item.provider and recommended.model_id == item.model_id),
                install_strategy=item.install_strategy,
            )
            for item in descriptors
        ]

        return ModelCatalogResponse(
            source_lang=source,
            target_lang=target,
            recommended_provider=recommended.provider if recommended else None,
            recommended_model_id=recommended.model_id if recommended else None,
            profile=profile.to_response(),
            models=models,
        )

    def recommend(
        self,
        source_lang: str,
        target_lang: str,
        profile: SystemProfile,
        descriptors: list[ModelDescriptor] | None = None,
    ) -> ModelDescriptor | None:
        available = [item for item in (descriptors or self.list_models(source_lang, target_lang, profile).models) if item.available]
        if not available:
            return None

        def pick(provider: str) -> ModelDescriptor | None:
            for item in available:
                if item.provider == provider:
                    return item
            return None

        if profile.tier == "low":
            return pick("argos") or pick("marian") or pick("nllb")
        if profile.tier == "standard":
            return pick("marian") or pick("argos") or pick("nllb")
        return pick("marian") or pick("nllb") or pick("argos")

    def validate_processing_target(
        self,
        profile: SystemProfile,
        provider: str,
        model_id: str,
        source_lang: str,
        target_lang: str,
        processing_device: str,
    ) -> tuple[bool, str | None]:
        response = self.list_models(source_lang, target_lang, profile)
        model = next(
            (item for item in response.models if item.provider == provider and item.model_id == model_id),
            None,
        )
        if model is None:
            return False, "Requested model is not available for this language pair."
        if not model.available:
            return False, model.availability_reason or "Requested model is unavailable."
        normalized = (processing_device or "auto").lower()
        if normalized == "auto":
            return True, None
        if normalized not in model.supported_targets:
            supported = ", ".join(model.supported_targets) or "none"
            return False, f"{model.label} does not support '{normalized}'. Supported targets: {supported}."
        if normalized != "cpu" and not profile.supports_target(normalized):
            return False, f"The current machine does not expose the '{normalized}' processing target."
        return True, None

    def _argos_descriptors(self, source_lang: str, target_lang: str) -> list[ModelDescriptor]:
        source = get_language(source_lang)
        target = get_language(target_lang)
        if not source or not target or not source.iso639_1 or not target.iso639_1:
            return []

        pairs = self._load_argos_index()
        direct_key = (source.iso639_1, target.iso639_1)
        pivot_source_key = (source.iso639_1, "en")
        pivot_target_key = ("en", target.iso639_1)

        direct = pairs.get(direct_key)
        if direct:
            model_id = f"argos:{source.iso639_1}-{target.iso639_1}"
            return [
                ModelDescriptor(
                    provider="argos",
                    model_id=model_id,
                    label=f"Argos Translate ({source.iso639_1} -> {target.iso639_1})",
                    source_lang=source_lang,
                    target_lang=target_lang,
                    size_mb=direct,
                    available=True,
                    direct=True,
                    installed=self.registry.is_installed("argos", model_id),
                    installed_targets=("cpu",) if self.registry.is_installed("argos", model_id) else (),
                    supported_targets=("cpu",),
                    recommended_target="cpu",
                    availability_reason="Direct Argos package available.",
                    note="Smallest offline option.",
                    license="MIT",
                )
            ]

        if pairs.get(pivot_source_key) and pairs.get(pivot_target_key):
            model_id = f"argos:{source.iso639_1}-en+en-{target.iso639_1}"
            size_mb = pairs[pivot_source_key] + pairs[pivot_target_key]
            return [
                ModelDescriptor(
                    provider="argos",
                    model_id=model_id,
                    label=f"Argos Translate pivot ({source.iso639_1} -> en -> {target.iso639_1})",
                    source_lang=source_lang,
                    target_lang=target_lang,
                    size_mb=size_mb,
                    available=True,
                    direct=False,
                    installed=self.registry.is_installed("argos", model_id),
                    installed_targets=("cpu",) if self.registry.is_installed("argos", model_id) else (),
                    supported_targets=("cpu",),
                    recommended_target="cpu",
                    availability_reason="Argos pair available through an English pivot.",
                    note="Uses English pivot because no direct Argos pair is indexed.",
                    license="MIT",
                    install_strategy="pivot",
                )
            ]

        return [
            ModelDescriptor(
                provider="argos",
                model_id=f"argos:{source.iso639_1}-{target.iso639_1}",
                label=f"Argos Translate ({source.iso639_1} -> {target.iso639_1})",
                source_lang=source_lang,
                target_lang=target_lang,
                size_mb=0,
                available=False,
                direct=True,
                installed=False,
                installed_targets=(),
                supported_targets=("cpu",),
                recommended_target="cpu",
                availability_reason="No direct or pivot package found in the Argos index.",
                note="No direct or pivot package found in the Argos index.",
                license="MIT",
            )
        ]

    def _marian_descriptors(self, source_lang: str, target_lang: str, profile: SystemProfile) -> list[ModelDescriptor]:
        source = get_language(source_lang)
        target = get_language(target_lang)
        if not source or not target or not source.marian_code or not target.marian_code:
            return []

        model_id = f"Helsinki-NLP/opus-mt-{source.marian_code}-{target.marian_code}"
        available = self._probe_huggingface_model(model_id)
        direct = True
        note = "CPU-friendly default when a direct pair exists."
        if not available and source_lang == "eng" and target_lang in MARIAN_ENGLISH_FALLBACKS:
            model_id, note = MARIAN_ENGLISH_FALLBACKS[target_lang]
            available = self._probe_huggingface_model(model_id)
            direct = False
        supported_targets = self._marian_supported_targets(profile)
        recommended_target = self._preferred_accelerated_target(profile, supported_targets)
        installed_targets = self._installed_targets_for_marian(model_id)
        return [
            ModelDescriptor(
                provider="marian",
                model_id=model_id,
                label=f"OPUS-MT / MarianMT ({source.marian_code} -> {target.marian_code})",
                source_lang=source_lang,
                target_lang=target_lang,
                size_mb=320,
                available=available,
                direct=direct,
                installed=bool(installed_targets),
                installed_targets=installed_targets,
                supported_targets=supported_targets,
                recommended_target=recommended_target,
                availability_reason="MarianMT supports this language pair." if available else "No MarianMT route was found for this language pair.",
                note=note,
                license="Apache-2.0",
            )
        ]

    def _nllb_descriptors(self, source_lang: str, target_lang: str, profile: SystemProfile) -> list[ModelDescriptor]:
        source = get_language(source_lang)
        target = get_language(target_lang)
        available = bool(source and target and source.nllb_code and target.nllb_code)
        installed_targets = self._installed_targets_for_nllb(NLLB_MODEL_ID)
        supported_targets = tuple(self._nllb_supported_targets(source_lang, target_lang, profile))
        return [
            ModelDescriptor(
                provider="nllb",
                model_id=NLLB_MODEL_ID,
                label="NLLB-200 distilled 600M",
                source_lang=source_lang,
                target_lang=target_lang,
                size_mb=NLLB_SIZE_MB,
                available=available,
                direct=True,
                installed=bool(installed_targets),
                installed_targets=installed_targets,
                supported_targets=supported_targets,
                recommended_target="cuda" if "cuda" in supported_targets else "cpu",
                availability_reason="NLLB supports this language pair." if available else "NLLB language codes are missing for this pair.",
                note="Universal multilingual fallback. Larger and slower on CPU.",
                license="CC-BY-NC-4.0",
            )
        ]

    def _load_argos_index(self) -> dict[tuple[str, str], int]:
        cached = self._read_metadata_cache()
        argos_pairs = cached.get("argos_pairs")
        if isinstance(argos_pairs, dict) and argos_pairs:
            return {(key.split("->")[0], key.split("->")[1]): value for key, value in argos_pairs.items()}

        pairs = self._fetch_argos_index()
        self._write_metadata_cache(
            {
                **cached,
                "argos_pairs": {f"{src}->{tgt}": size for (src, tgt), size in pairs.items()},
            }
        )
        return pairs

    def _fetch_argos_index(self) -> dict[tuple[str, str], int]:
        try:
            response = httpx.get(ARGOS_INDEX_URL, timeout=10.0)
            response.raise_for_status()
            payload = response.json()
            pairs: dict[tuple[str, str], int] = {}
            for item in payload:
                from_code = item.get("from_code")
                to_code = item.get("to_code")
                size = int(round((item.get("package_size", 0) or 0) / (1024 ** 2)))
                if from_code and to_code:
                    pairs[(from_code, to_code)] = size or 80
            if pairs:
                return pairs
        except Exception:
            pass

        # Fallback keeps the install UI useful when the remote index is unavailable.
        return {
            ("en", "es"): 82,
            ("es", "en"): 82,
            ("en", "fr"): 62,
            ("fr", "en"): 62,
            ("en", "de"): 70,
            ("de", "en"): 72,
            ("en", "ru"): 187,
            ("ru", "en"): 176,
            ("en", "it"): 78,
            ("it", "en"): 79,
            ("en", "pt"): 89,
            ("pt", "en"): 90,
            ("en", "tr"): 101,
            ("tr", "en"): 101,
        }

    def _probe_huggingface_model(self, model_id: str) -> bool:
        cached = self._read_metadata_cache()
        probe_map = cached.get("hf_models")
        if isinstance(probe_map, dict) and model_id in probe_map:
            return bool(probe_map[model_id])

        try:
            response = httpx.get(f"https://huggingface.co/api/models/{model_id}", timeout=10.0)
            exists = response.status_code == 200
        except Exception:
            exists = False

        probe_map = probe_map if isinstance(probe_map, dict) else {}
        probe_map[model_id] = exists
        self._write_metadata_cache({**cached, "hf_models": probe_map})
        return exists

    def _read_metadata_cache(self) -> dict[str, object]:
        if not self.metadata_cache_file.exists():
            return {}
        return json.loads(self.metadata_cache_file.read_text(encoding="utf-8"))

    def _write_metadata_cache(self, data: dict[str, object]) -> None:
        self.metadata_cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _is_effectively_installed(self, provider: str, model_id: str) -> bool:
        if self.registry.is_installed(provider, model_id):
            return True
        if provider in {"marian", "nllb"}:
            return hf_model_cache_exists(model_id)
        return False

    def _marian_supported_targets(self, profile: SystemProfile) -> tuple[str, ...]:
        targets = ["cpu"]
        if profile.supports_target("cuda"):
            targets.append("cuda")
        if profile.supports_target("openvino_gpu"):
            targets.append("openvino_gpu")
        if profile.supports_target("directml"):
            targets.append("directml")
        return tuple(targets)

    def _nllb_supported_targets(self, source_lang: str, target_lang: str, profile: SystemProfile) -> list[str]:
        source = get_language(source_lang)
        target = get_language(target_lang)
        if not source or not target or not source.nllb_code or not target.nllb_code:
            return []
        targets = ["cpu"]
        if profile.supports_target("cuda"):
            targets.append("cuda")
        return targets

    @staticmethod
    def _preferred_accelerated_target(profile: SystemProfile, supported_targets: tuple[str, ...]) -> str:
        for target in ("cuda", "openvino_gpu", "directml", "cpu"):
            if target in supported_targets and (target == "cpu" or profile.supports_target(target)):
                return target
        return "cpu"

    def _installed_targets_for_marian(self, model_id: str) -> tuple[str, ...]:
        installed = set(self.registry.installed_targets("marian", model_id))
        if self.registry.is_installed("marian", model_id) or hf_model_cache_exists(model_id):
            installed.update({"cpu", "cuda"})
        if local_model_artifact_exists(self.model_artifacts_dir, "marian", model_id, "directml"):
            installed.add("directml")
        if local_model_artifact_exists(self.model_artifacts_dir, "marian", model_id, "openvino_gpu"):
            installed.add("openvino_gpu")
        return tuple(sorted(installed))

    def _installed_targets_for_nllb(self, model_id: str) -> tuple[str, ...]:
        installed = set(self.registry.installed_targets("nllb", model_id))
        if self.registry.is_installed("nllb", model_id) or hf_model_cache_exists(model_id):
            installed.update({"cpu", "cuda"})
        return tuple(sorted(installed))
