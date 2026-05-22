from __future__ import annotations

import json
import sys

from polylinguist.services.languages import get_language
from polylinguist.services.translation import (
    NLLB_BATCH_SIZE,
    TranslationError,
    _argos_path_from_model_id,
    _hf_runtime_packages,
    _prepare_marian_batch,
    _resolve_device_preference,
)


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    payload = json.loads(sys.stdin.read() or "{}")
    try:
        result = dispatch(command, payload)
    except Exception as exc:
        print(json.dumps({"event": "result", "ok": False, "error": str(exc)}), flush=True)
        raise SystemExit(1)
    print(json.dumps({"event": "result", "ok": True, "result": result}), flush=True)


def dispatch(command: str, payload: dict[str, object]) -> object:
    if command == "install-argos":
        return install_argos(str(payload["model_id"]), str(payload.get("install_strategy", "direct")))
    if command == "install-hf":
        return install_hf(str(payload["model_id"]))
    if command == "translate-argos":
        return translate_argos(
            str(payload["model_id"]),
            list(payload.get("cues", [])),
        )
    if command == "translate-marian":
        return translate_marian(
            str(payload["model_id"]),
            str(payload.get("target_lang", "")),
            list(payload.get("cues", [])),
            str(payload.get("device_preference", "auto")),
        )
    if command == "translate-nllb":
        return translate_nllb(
            str(payload["model_id"]),
            str(payload["source_lang"]),
            str(payload["target_lang"]),
            list(payload.get("cues", [])),
            str(payload.get("device_preference", "auto")),
        )
    raise TranslationError(f"Unknown worker command: {command}")


def install_argos(model_id: str, install_strategy: str) -> str:
    import argostranslate.package  # type: ignore

    source, target = _argos_path_from_model_id(model_id)
    package_specs = [(source, target)] if install_strategy == "direct" else [(source, "en"), ("en", target)]
    emit("index", "Refreshing Argos package index.")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    installed_pairs: list[str] = []
    for from_code, to_code in package_specs:
        emit("download", f"Downloading Argos package {from_code}->{to_code}.")
        match = next(
            (pkg for pkg in available if pkg.from_code == from_code and pkg.to_code == to_code),
            None,
        )
        if not match:
            raise TranslationError(f"Missing Argos package for {from_code}->{to_code}.")
        package_path = match.download()
        emit("install", f"Installing Argos package {from_code}->{to_code}.")
        argostranslate.package.install_from_path(package_path)
        installed_pairs.append(f"{from_code}->{to_code}")
    return f"Installed Argos packages: {', '.join(installed_pairs)}"


def install_hf(model_id: str) -> str:
    from huggingface_hub import snapshot_download  # type: ignore

    emit("download", f"Downloading model weights for {model_id}.")
    location = snapshot_download(repo_id=model_id)
    return f"Downloaded {model_id} to {location}"


def translate_argos(model_id: str, cues: list[str]) -> list[str]:
    import argostranslate.translate  # type: ignore

    source, target = _argos_path_from_model_id(model_id)
    translations: list[str] = []
    total = len(cues)
    for index, text in enumerate(cues, start=1):
        emit("translate", f"Translating cue {index}/{total}.")
        if "+en-" in model_id:
            translated = argostranslate.translate.translate(
                argostranslate.translate.translate(text, source, "en"),
                "en",
                target,
            )
        else:
            translated = argostranslate.translate.translate(text, source, target)
        translations.append(translated)
    return translations


def translate_marian(model_id: str, target_lang: str, cues: list[str], device_preference: str) -> list[str]:
    import torch  # type: ignore
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    if not cues:
        return []
    emit("load", f"Loading tokenizer for {model_id}.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    emit("load", f"Loading model weights for {model_id}.")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    device = _resolve_device_preference(device_preference)
    if device != "cpu":
        model = model.to(device)
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
        model.generation_config.max_length = None
    emit("load", f"Model loaded on {device}.")
    batch_size = 8
    total_batches = (len(cues) + batch_size - 1) // batch_size
    translations: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(cues), batch_size), start=1):
            batch = cues[start : start + batch_size]
            batch = _prepare_marian_batch(model_id, target_lang, batch)
            emit("translate", f"Translating batch {batch_index}/{total_batches}.")
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            if device != "cpu":
                encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(**encoded, max_new_tokens=256)
            translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return translations


def translate_nllb(model_id: str, source_lang: str, target_lang: str, cues: list[str], device_preference: str) -> list[str]:
    import torch  # type: ignore
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    source = get_language(source_lang)
    target = get_language(target_lang)
    if not source or not target or not source.nllb_code or not target.nllb_code:
        raise TranslationError("Requested language pair is not available in NLLB.")
    device = _resolve_device_preference(device_preference)
    emit("load", f"Loading tokenizer for {model_id}.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    emit("load", f"Loading model weights for {model_id}.")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if device != "cpu":
        model = model.to(device)
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
        model.generation_config.max_length = None
    emit("load", f"Model loaded on {device}.")
    total_batches = (len(cues) + NLLB_BATCH_SIZE - 1) // NLLB_BATCH_SIZE
    translations: list[str] = []
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target.nllb_code)
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(cues), NLLB_BATCH_SIZE), start=1):
            emit("translate", f"Translating batch {batch_index}/{total_batches}.")
            batch = cues[start : start + NLLB_BATCH_SIZE]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            if device != "cpu":
                encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
            )
            translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return translations


def emit(stage: str, message: str) -> None:
    print(json.dumps({"event": "progress", "stage": stage, "message": message}), flush=True)


if __name__ == "__main__":
    main()
