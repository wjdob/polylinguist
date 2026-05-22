from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polylinguist.services.languages import get_language
from polylinguist.services.subtitles import parse_subtitle_text


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    source_lang = get_language(args.source_lang)
    target_lang = get_language(args.target_lang)
    if source_lang is None or target_lang is None:
        raise SystemExit("Unsupported source or target language.")

    text = Path(args.subtitle).read_text(encoding="utf-8-sig", errors="replace")
    cues = parse_subtitle_text(text)
    cue_texts = [cue.text for cue in cues if cue.text.strip()]
    if args.limit:
        cue_texts = cue_texts[: args.limit]
    char_count = sum(len(item) for item in cue_texts)

    started = time.perf_counter()
    if args.provider == "argos":
        result = benchmark_argos(args, source_lang.iso639_1, target_lang.iso639_1, cue_texts)
    else:
        result = benchmark_hf(args, source_lang.nllb_code, target_lang.nllb_code, cue_texts)
    total_seconds = time.perf_counter() - started

    output = {
        "provider": args.provider,
        "model": result["model"],
        "requested_device": args.device,
        "effective_device": result["effective_device"],
        "dtype": result.get("dtype"),
        "subtitle": str(Path(args.subtitle)),
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "cue_count": len(cue_texts),
        "char_count": char_count,
        "batch_size": args.batch_size,
        "target_token": result.get("target_token"),
        "setup_seconds": round(result["setup_seconds"], 3),
        "load_seconds": round(result["load_seconds"], 3),
        "translation_seconds": round(result["translation_seconds"], 3),
        "total_seconds": round(total_seconds, 3),
        "cues_per_second": round(len(cue_texts) / result["translation_seconds"], 3)
        if result["translation_seconds"]
        else None,
        "chars_per_second": round(char_count / result["translation_seconds"], 3)
        if result["translation_seconds"]
        else None,
        "gpu_memory_mb": result.get("gpu_memory_mb"),
        "sample_output": result["sample_output"],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


def benchmark_argos(args: argparse.Namespace, source: str | None, target: str | None, cues: list[str]) -> dict[str, Any]:
    if source is None or target is None:
        raise SystemExit("Argos requires ISO-639-1 language codes for this pair.")
    setup_started = time.perf_counter()
    import argostranslate.package  # type: ignore
    import argostranslate.translate  # type: ignore

    if args.ensure_models:
        ensure_argos_package(argostranslate.package, argostranslate.translate, source, target)
    setup_seconds = time.perf_counter() - setup_started

    translate_started = time.perf_counter()
    translations = [argostranslate.translate.translate(cue, source, target) for cue in cues]
    translation_seconds = time.perf_counter() - translate_started
    return {
        "model": f"argos:{source}-{target}",
        "effective_device": "cpu",
        "setup_seconds": setup_seconds,
        "load_seconds": 0.0,
        "translation_seconds": translation_seconds,
        "sample_output": translations[:3],
    }


def ensure_argos_package(package_module: Any, translate_module: Any, source: str, target: str) -> None:
    for language in translate_module.get_installed_languages():
        if language.code == source and any(translation.to_lang.code == target for translation in language.translations_to):
            return

    package_module.update_package_index()
    available = package_module.get_available_packages()
    match = next(
        (pkg for pkg in available if pkg.from_code == source and pkg.to_code == target),
        None,
    )
    if match is None:
        raise SystemExit(f"No Argos package is available for {source}->{target}.")
    package_path = match.download()
    package_module.install_from_path(package_path)


def benchmark_hf(
    args: argparse.Namespace,
    source_nllb: str | None,
    target_nllb: str | None,
    cues: list[str],
) -> dict[str, Any]:
    setup_started = time.perf_counter()
    import torch  # type: ignore
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    device = "cuda" if args.device == "cuda" else "cpu"
    dtype = resolve_dtype(args, torch, device)
    setup_seconds = time.perf_counter() - setup_started

    model_id = args.model_id or (
        "facebook/nllb-200-distilled-600M"
        if args.provider == "nllb"
        else "Helsinki-NLP/opus-mt-en-ine"
    )
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if args.provider == "nllb":
        if source_nllb is None or target_nllb is None:
            raise SystemExit("NLLB language codes are missing for this pair.")
        tokenizer.src_lang = source_nllb
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if args.low_cpu_mem_usage:
        model_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **model_kwargs)
    model = model.to(device)
    model.eval()
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
        model.generation_config.max_length = None
    load_seconds = time.perf_counter() - load_started

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    translate_started = time.perf_counter()
    target_token = resolve_marian_target_token(args, model_id)
    translations: list[str] = []
    with torch.no_grad():
        for start in range(0, len(cues), args.batch_size):
            batch = cues[start : start + args.batch_size]
            if target_token:
                batch = [f">>{target_token}<< {item}" for item in batch]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generate_kwargs: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
            if args.provider == "nllb":
                generate_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(target_nllb)
            generated = model.generate(**encoded, **generate_kwargs)
            translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    if device == "cuda":
        torch.cuda.synchronize()
    translation_seconds = time.perf_counter() - translate_started
    gpu_memory_mb = None
    if device == "cuda":
        gpu_memory_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    return {
        "model": model_id,
        "effective_device": device,
        "dtype": str(dtype).replace("torch.", "") if dtype is not None else "default",
        "setup_seconds": setup_seconds,
        "load_seconds": load_seconds,
        "translation_seconds": translation_seconds,
        "gpu_memory_mb": gpu_memory_mb,
        "sample_output": translations[:3],
        "target_token": target_token,
    }


def resolve_dtype(args: argparse.Namespace, torch_module: Any, device: str) -> Any:
    if args.dtype == "fp32":
        return torch_module.float32
    if args.dtype == "fp16":
        return torch_module.float16
    if args.dtype == "auto" and device == "cuda" and args.provider == "nllb":
        return torch_module.float16
    return None


def resolve_marian_target_token(args: argparse.Namespace, model_id: str) -> str | None:
    if args.provider != "marian":
        return None
    if args.target_token:
        return args.target_token
    if "opus-mt-en-ine" in model_id or "opus-mt-en-mul" in model_id:
        target = get_language(args.target_lang)
        return target.canonical if target else args.target_lang
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtitle", required=True)
    parser.add_argument("--provider", choices=["argos", "marian", "nllb"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--source-lang", default="eng")
    parser.add_argument("--target-lang", default="pol")
    parser.add_argument("--model-id")
    parser.add_argument("--target-token")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ensure-models", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
