from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock, Thread
from typing import Callable
import uuid


InstallFn = Callable[[Callable[[str, str], None]], str]


@dataclass
class InstallJob:
    job_id: str
    provider: str
    model_id: str
    status: str = "queued"
    stage: str = "queued"
    message: str = "Waiting to start."
    detail: str | None = None
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=80))

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "detail": self.detail,
            "log_lines": list(self.log_lines),
        }


class InstallJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, InstallJob] = {}
        self._lock = Lock()

    def create_job(self, provider: str, model_id: str, install_fn: InstallFn) -> InstallJob:
        job = InstallJob(
            job_id=uuid.uuid4().hex,
            provider=provider,
            model_id=model_id,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        thread = Thread(target=self._run_job, args=(job.job_id, install_fn), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> InstallJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run_job(self, job_id: str, install_fn: InstallFn) -> None:
        def progress(stage: str, message: str) -> None:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "running"
                job.stage = stage
                job.message = message
                job.log_lines.append(f"[{stage}] {message}")

        progress("starting", "Preparing installation job.")
        try:
            detail = install_fn(progress)
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.stage = "failed"
                job.message = str(exc)
                job.detail = str(exc)
                job.log_lines.append(f"[failed] {exc}")
            return

        with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.stage = "completed"
            job.message = "Installation finished."
            job.detail = detail
            job.log_lines.append("[completed] Installation finished.")
