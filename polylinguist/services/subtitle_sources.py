from __future__ import annotations

from dataclasses import dataclass
from os.path import basename
from typing import Any
from urllib.parse import urlencode

import httpx

from polylinguist.services.languages import normalize_language
from polylinguist.services.subtitles import SubtitleCandidate


@dataclass(frozen=True)
class SubtitleSourceExtra:
    filename: str | None = None
    video_size: str | None = None
    video_hash: str | None = None


class OpenSubtitlesProvider:
    base_url = "https://opensubtitles-v3.strem.io"

    async def list_candidates(
        self,
        media_type: str,
        media_id: str,
        source_lang: str,
        extra: SubtitleSourceExtra,
        limit: int = 3,
    ) -> list[SubtitleCandidate]:
        data = await self._fetch_subtitles(media_type, media_id, extra)
        normalized_lang = normalize_language(source_lang)
        items = [item for item in data if normalize_language(item.get("lang")) == normalized_lang]
        ranked = sorted(items, key=self._score, reverse=True)
        candidates: list[SubtitleCandidate] = []
        for item in ranked[:limit]:
            candidates.append(
                SubtitleCandidate(
                    subtitle_id=str(item.get("id")),
                    url=item["url"],
                    lang=normalize_language(item.get("lang")) or source_lang,
                    label=self._candidate_name(item),
                    source="opensubtitles",
                    score=self._score(item),
                    format=self._guess_format(item.get("url", "")),
                    match_label=self._match_label(item),
                )
            )
        return candidates

    async def fetch_subtitle_text(self, candidate: SubtitleCandidate) -> str:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(candidate.url)
            response.raise_for_status()
            return response.text

    async def _fetch_subtitles(
        self,
        media_type: str,
        media_id: str,
        extra: SubtitleSourceExtra,
    ) -> list[dict[str, Any]]:
        path = f"/subtitles/{media_type}/{media_id}"
        extra_params: dict[str, str] = {}
        if extra.filename:
            extra_params["filename"] = extra.filename
        if extra.video_size:
            extra_params["videoSize"] = extra.video_size
        if extra.video_hash:
            extra_params["videoHash"] = extra.video_hash
        if extra_params:
            path += "/" + urlencode(extra_params)
        path += ".json"

        async with httpx.AsyncClient(base_url=self.base_url, timeout=15.0, follow_redirects=True) as client:
            response = await client.get(path)
            response.raise_for_status()
            payload = response.json()
            return payload.get("subtitles", [])

    @staticmethod
    def _score(item: dict[str, Any]) -> float:
        score = 0.0
        if item.get("m") == "i":
            score += 100.0
        if item.get("g"):
            score += 20.0
        score += float(item.get("downloads") or 0) / 1000.0
        return score

    @staticmethod
    def _match_label(item: dict[str, Any]) -> str:
        if item.get("m") == "i":
            return "hash match"
        if item.get("g"):
            return "release match"
        return "alternate"

    @staticmethod
    def _guess_format(url: str) -> str:
        lowered = url.lower()
        if lowered.endswith(".vtt"):
            return "vtt"
        return "srt"

    @staticmethod
    def _candidate_name(item: dict[str, Any]) -> str:
        for key in ("subfilename", "subFileName", "SubFileName", "releaseInfo", "release", "name", "label"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        url = str(item.get("url") or "")
        if url:
            return basename(url.split("?", 1)[0]) or "OpenSubtitles candidate"
        return "OpenSubtitles candidate"
