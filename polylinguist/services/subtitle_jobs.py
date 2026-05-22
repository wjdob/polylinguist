from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from threading import Lock


PROGRESS_RE = re.compile(
    r"(?:(?:batch|cue)\s+(?P<current>\d+)\s*/\s*(?P<total>\d+))",
    re.IGNORECASE,
)


@dataclass
class SubtitleGenerationJob:
    cache_key: str
    status: str = "queued"
    stage: str = "queued"
    message: str = "Waiting to start."
    detail: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    progress_percent: int | None = None
    source_subtitle_name: str | None = None
    source_subtitle_id: str | None = None
    source_subtitle_url: str | None = None
    media_filename: str | None = None
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=100))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_key": self.cache_key,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "detail": self.detail,
            "progress_current": self.progress_current,
            "progress_total": self.progress_total,
            "progress_percent": self.progress_percent,
            "source_subtitle_name": self.source_subtitle_name,
            "source_subtitle_id": self.source_subtitle_id,
            "source_subtitle_url": self.source_subtitle_url,
            "media_filename": self.media_filename,
            "log_lines": list(self.log_lines),
            "updated_at": self.updated_at.isoformat(),
        }


class SubtitleGenerationTracker:
    def __init__(self) -> None:
        self._jobs: dict[str, SubtitleGenerationJob] = {}
        self._lock = Lock()

    def get(self, cache_key: str) -> SubtitleGenerationJob | None:
        with self._lock:
            return self._jobs.get(cache_key)

    def recent(self, limit: int = 10) -> list[SubtitleGenerationJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.updated_at, reverse=True)[:limit]

    def describe(
        self,
        cache_key: str,
        *,
        source_subtitle_name: str | None = None,
        source_subtitle_id: str | None = None,
        source_subtitle_url: str | None = None,
        media_filename: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.setdefault(cache_key, SubtitleGenerationJob(cache_key=cache_key))
            if source_subtitle_name:
                job.source_subtitle_name = source_subtitle_name
            if source_subtitle_id:
                job.source_subtitle_id = source_subtitle_id
            if source_subtitle_url:
                job.source_subtitle_url = source_subtitle_url
            if media_filename:
                job.media_filename = media_filename
            job.updated_at = datetime.now(timezone.utc)

    def progress(self, cache_key: str, stage: str, message: str, status: str = "running") -> None:
        with self._lock:
            job = self._jobs.setdefault(cache_key, SubtitleGenerationJob(cache_key=cache_key))
            job.status = status
            job.stage = stage
            job.message = message
            match = PROGRESS_RE.search(message)
            if match:
                current = int(match.group("current"))
                total = max(int(match.group("total")), 1)
                job.progress_current = current
                job.progress_total = total
                job.progress_percent = max(0, min(100, int((current / total) * 100)))
            job.updated_at = datetime.now(timezone.utc)
            job.log_lines.append(f"[{stage}] {message}")

    def completed(self, cache_key: str, detail: str | None = None) -> None:
        with self._lock:
            job = self._jobs.setdefault(cache_key, SubtitleGenerationJob(cache_key=cache_key))
            job.status = "completed"
            job.stage = "completed"
            job.message = "Subtitle is ready."
            job.detail = detail
            job.progress_current = job.progress_total or job.progress_current
            if job.progress_total:
                job.progress_percent = 100
            job.updated_at = datetime.now(timezone.utc)
            job.log_lines.append("[completed] Subtitle is ready.")

    def failed(self, cache_key: str, error: Exception) -> None:
        with self._lock:
            job = self._jobs.setdefault(cache_key, SubtitleGenerationJob(cache_key=cache_key))
            job.status = "failed"
            job.stage = "failed"
            job.message = str(error)
            job.detail = str(error)
            job.updated_at = datetime.now(timezone.utc)
            job.log_lines.append(f"[failed] {error}")

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()
