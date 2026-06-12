from __future__ import annotations

import base64
import html
import json
import re
from dataclasses import dataclass

from polylinguist.schemas import AddonSettings
from polylinguist.services.languages import language_label


_TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
)
_PUNCTUATION_ONLY_RE = re.compile(r"^[\s,.;:!?_\-+=~`'\"()\[\]{}<>\\/|*]+$")


@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class SubtitleCandidate:
    subtitle_id: str
    url: str
    lang: str
    label: str
    source: str
    score: float
    format: str = "srt"
    match_label: str = "match"


@dataclass(frozen=True)
class PreparedTranslationBatch:
    cues: list[str]
    active_indices: list[int]
    active_cues: list[str]
    sanitized_count: int
    skipped_count: int


def parse_subtitle_text(content: str) -> list[SubtitleCue]:
    stripped = content.lstrip()
    if stripped.startswith("WEBVTT"):
        return _parse_vtt(content)
    return _parse_srt(content)


def render_dual_srt(
    cues: list[SubtitleCue],
    translations: list[str],
    settings: AddonSettings,
) -> str:
    lines: list[str] = []
    dual_mode = settings.format_mode == "dual"
    for index, cue in enumerate(cues, start=1):
        primary = _normalize_cue_text(cue.text)
        translated = _normalize_cue_text(translations[index - 1]) if index - 1 < len(translations) else ""
        if not primary and not translated:
            continue
        block_lines = [str(index), f"{_to_timestamp(cue.start_ms)} --> {_to_timestamp(cue.end_ms)}"]
        if dual_mode:
            if primary:
                block_lines.append(primary)
            if translated:
                block_lines.append(translated)
        else:
            block_lines.append(translated or primary)
        lines.append("\r\n".join(block_lines))
    return "\r\n\r\n".join(lines).strip() + "\r\n"


def subtitle_menu_label(candidate: SubtitleCandidate, settings: AddonSettings, provider_label: str, index: int) -> str:
    src = language_label(settings.source_lang)
    tgt = language_label(settings.target_lang)
    return f"Dual {src}+{tgt} - {provider_label} - {candidate.match_label} #{index}"


def prepare_translation_batch(texts: list[str]) -> PreparedTranslationBatch:
    cues: list[str] = []
    active_indices: list[int] = []
    active_cues: list[str] = []
    sanitized_count = 0
    skipped_count = 0
    for index, text in enumerate(texts):
        prepared, changed = _prepare_translation_text(text)
        if changed:
            sanitized_count += 1
        cues.append(prepared)
        if prepared:
            active_indices.append(index)
            active_cues.append(prepared)
        else:
            skipped_count += 1
    return PreparedTranslationBatch(
        cues=cues,
        active_indices=active_indices,
        active_cues=active_cues,
        sanitized_count=sanitized_count,
        skipped_count=skipped_count,
    )


def encode_subtitle_payload(payload: dict[str, str]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_subtitle_payload(value: str) -> dict[str, str]:
    padding = "=" * (-len(value) % 4)
    decoded = base64.urlsafe_b64decode(value + padding)
    return json.loads(decoded.decode("utf-8"))


def _parse_srt(content: str) -> list[SubtitleCue]:
    blocks = re.split(r"\r?\n\r?\n", content.strip())
    cues: list[SubtitleCue] = []
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) >= 2 and lines[0].isdigit():
            timestamp_line = lines[1]
            text_lines = lines[2:]
            index = int(lines[0])
        else:
            timestamp_line = lines[0]
            text_lines = lines[1:]
            index = len(cues) + 1
        match = _TIMESTAMP_RE.search(timestamp_line)
        if not match:
            continue
        cues.append(
            SubtitleCue(
                index=index,
                start_ms=_from_timestamp(match.group("start")),
                end_ms=_from_timestamp(match.group("end")),
                text="\n".join(text_lines).strip(),
            )
        )
    return cues


def _parse_vtt(content: str) -> list[SubtitleCue]:
    lines = content.splitlines()
    cues: list[SubtitleCue] = []
    idx = 0
    cue_index = 1
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line == "WEBVTT":
            idx += 1
            continue
        if "-->" not in line and idx + 1 < len(lines) and "-->" in lines[idx + 1]:
            idx += 1
            line = lines[idx].strip()
        match = _TIMESTAMP_RE.search(line.replace(".", ","))
        if not match:
            idx += 1
            continue
        idx += 1
        text_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip():
            text_lines.append(lines[idx])
            idx += 1
        cues.append(
            SubtitleCue(
                index=cue_index,
                start_ms=_from_timestamp(match.group("start")),
                end_ms=_from_timestamp(match.group("end")),
                text="\n".join(text_lines).strip(),
            )
        )
        cue_index += 1
        idx += 1
    return cues


def _normalize_cue_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = re.sub(r"\{[^}]+\}", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned.replace("\n", " ")).strip()
    return html.escape(cleaned, quote=False)


def _prepare_translation_text(text: str) -> tuple[str, bool]:
    if not text:
        return "", False
    changed = False
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        cleaned = re.sub(r"<[^>]+>", "", stripped)
        cleaned = re.sub(r"\{[^}]+\}", "", cleaned).strip()
        if cleaned != stripped:
            changed = True
        if not cleaned:
            if stripped:
                changed = True
            continue
        if len(cleaned) >= 12 and _PUNCTUATION_ONLY_RE.fullmatch(cleaned):
            changed = True
            continue
        lines.append(cleaned)
    prepared = "\n".join(lines).strip()
    if prepared != text.strip():
        changed = True
    return prepared, changed


def _from_timestamp(timestamp: str) -> int:
    normalized = timestamp.replace(".", ",")
    hours, minutes, seconds_ms = normalized.split(":")
    seconds, milliseconds = seconds_ms.split(",")
    return (
        int(hours) * 3600 * 1000
        + int(minutes) * 60 * 1000
        + int(seconds) * 1000
        + int(milliseconds)
    )


def _to_timestamp(value: int) -> str:
    hours = value // 3600000
    value %= 3600000
    minutes = value // 60000
    value %= 60000
    seconds = value // 1000
    milliseconds = value % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
