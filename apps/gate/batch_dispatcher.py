"""Serial headless batch job dispatcher for Comfy Colab Kit.

Replaces the legacy reverse-proxy tunnel with a compliant batch pipeline:
1. Accepts ComfyUI API graphs (`POST /prompt`) or high-level prompts (`POST /_gate/txt2img`)
   plus reference image uploads (`POST /upload/image`).
2. Uploads required reference images & `task.json` via `colab --auth oauth2 upload`.
3. Executes `headless_runner.py` inside the warm Colab Jupyter kernel (`colab --auth oauth2 exec`).
4. Downloads rendered images to local VPS storage (`colab --auth oauth2 download`),
   serves them via `GET /view` with strict path containment, and enforces rolling
   retention limits (`max_output_files` and `output_retention_days`).
5. Resets the session's keep-warm idle timer (`session.touch()`) on job start and completion.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import tempfile
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .colab import ColabError
from .config import Settings
from .session import Session, redact_secrets
from .state import State

log = logging.getLogger("gate.batch")

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


class BatchError(RuntimeError):
    def __init__(self, status: int, code: str, msg: str):
        super().__init__(msg)
        self.status, self.code, self.msg = status, code, msg


def sanitize_filename(name: str, default_stem: str = "image") -> str:
    decoded = urllib.parse.unquote(name or "")
    raw = os.path.basename(decoded.strip().replace("\\", "/"))
    stem, ext = os.path.splitext(raw)
    ext = ext.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise BatchError(400, "unsupported_file_type", f"Unsupported image extension {ext!r}; allowed: png, jpg, jpeg, webp")
    clean_stem = SAFE_NAME_RE.sub("_", stem).strip("._-")[:64] or default_stem
    return f"{clean_stem}{ext}"


def safe_resolve_under(base_dir: Path, filename: str, subfolder: str = "") -> Path:
    """Resolve `base_dir / subfolder / filename` and reject any directory traversal."""
    if not filename or "\x00" in filename or "/" in filename or "\\" in filename or filename.startswith("."):
        raise BatchError(400, "invalid_filename", "Invalid filename")
    if "\x00" in subfolder or ".." in subfolder.replace("\\", "/").split("/"):
        raise BatchError(400, "invalid_subfolder", "Invalid subfolder")
    base_resolved = base_dir.resolve()
    sub_clean = subfolder.strip().strip("/\\")
    candidate = (base_resolved / sub_clean / filename).resolve() if sub_clean else (base_resolved / filename).resolve()
    if not candidate.is_relative_to(base_resolved):
        raise BatchError(400, "path_traversal_blocked", "Path escapes storage directory")
    return candidate


def _collect_string_leaves(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_string_leaves(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_string_leaves(v, out)
    elif isinstance(obj, str) and len(obj) <= 160:
        out.add(obj)


@dataclass
class BatchJob:
    prompt_id: str
    number: int
    prompt: dict[str, Any]
    submitted_by: str
    created_at: float
    title: str = ""
    preset: str = ""
    timeout_s: int = 300
    status: str = "queued"  # queued | waiting_gpu | running | success | error | cancelled
    started_at: float | None = None
    finished_at: float | None = None
    duration_s: float | None = None
    step: int | None = None
    max_steps: int | None = None
    current_node: str | None = None
    error: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    done_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def to_summary(self) -> dict[str, Any]:
        images = []
        for node_id, node_out in self.outputs.items():
            for img in node_out.get("images") or []:
                fn = img.get("filename", "")
                images.append(
                    {
                        "filename": fn,
                        "node_id": node_id,
                        "url": f"/view?filename={fn}&type=output",
                    }
                )
        return {
            "prompt_id": self.prompt_id,
            "number": self.number,
            "title": self.title,
            "preset": self.preset,
            "submitted_by": self.submitted_by,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "step": self.step,
            "max_steps": self.max_steps,
            "error": self.error,
            "images": images,
        }

    def to_comfy_history_entry(self) -> dict[str, Any]:
        status_str = "success" if self.status == "success" else "error"
        msgs = []
        if self.error:
            msgs.append(["execution_error", {"exception_message": self.error}])
        return {
            "prompt": [self.number, self.prompt_id, self.prompt, {}, []],
            "outputs": self.outputs,
            "status": {
                "status_str": status_str,
                "completed": self.status in ("success", "error"),
                "messages": msgs,
            },
        }


class BatchDispatcher:
    def __init__(self, settings: Settings, session: Session):
        self.settings = settings
        self.session = session
        self.state_dir = Path(settings.state_dir)
        self.inputs_dir = self.state_dir / "inputs"
        self.outputs_dir = self.state_dir / "outputs"
        self.jobs_file = self.state_dir / "jobs.json"
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self._seq = 0
        self._queue: asyncio.Queue[BatchJob] = asyncio.Queue()
        self._jobs: dict[str, BatchJob] = {}
        self._running: BatchJob | None = None
        self._worker_task: asyncio.Task | None = None
        self._uploaded_inputs: dict[str, int] = {}  # filename -> mtime_ns for current VM session
        self._last_session_started: float | None = None
        self._restore_jobs()

    def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def close(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._persist_jobs()

    def _sync_busy_flag(self) -> None:
        self.session.queue_busy = bool(self._running is not None or not self._queue.empty())

    def _restore_jobs(self) -> None:
        if not self.jobs_file.is_file():
            return
        try:
            items = json.loads(self.jobs_file.read_text(encoding="utf-8"))
            if not isinstance(items, list):
                return
            for raw in items[-200:]:
                pid = str(raw.get("prompt_id") or "")
                if not pid:
                    continue
                num = int(raw.get("number") or 0)
                self._seq = max(self._seq, num)
                st = str(raw.get("status") or "error")
                if st in ("queued", "waiting_gpu", "running"):
                    st = "error"
                job = BatchJob(
                    prompt_id=pid,
                    number=num,
                    prompt=raw.get("prompt") if isinstance(raw.get("prompt"), dict) else {},
                    submitted_by=str(raw.get("submitted_by") or ""),
                    created_at=float(raw.get("created_at") or time.time()),
                    title=str(raw.get("title") or ""),
                    preset=str(raw.get("preset") or ""),
                    status=st,
                    started_at=raw.get("started_at"),
                    finished_at=raw.get("finished_at"),
                    duration_s=raw.get("duration_s"),
                    error=str(raw.get("error") or ("Interrupted by gateway restart" if st == "error" and not raw.get("error") else "")),
                    outputs=raw.get("outputs") if isinstance(raw.get("outputs"), dict) else {},
                )
                job.done_event.set()
                self._jobs[pid] = job
        except Exception as e:
            log.warning("failed to restore jobs.json: %s", e)

    def _persist_jobs(self) -> None:
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            ordered = sorted(self._jobs.values(), key=lambda j: j.created_at)[-200:]
            payload = [
                {
                    "prompt_id": j.prompt_id,
                    "number": j.number,
                    "prompt": j.prompt,
                    "submitted_by": j.submitted_by,
                    "created_at": j.created_at,
                    "title": j.title,
                    "preset": j.preset,
                    "status": j.status,
                    "started_at": j.started_at,
                    "finished_at": j.finished_at,
                    "duration_s": j.duration_s,
                    "error": j.error,
                    "outputs": j.outputs,
                }
                for j in ordered
            ]
            tmp = self.jobs_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.jobs_file)
        except Exception as e:
            log.warning("failed to persist jobs.json: %s", e)

    def save_input_image(self, filename: str, data: bytes, overwrite: bool = True) -> dict[str, Any]:
        if not data:
            raise BatchError(400, "empty_upload", "Uploaded image is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise BatchError(413, "upload_too_large", "Image exceeds 32 MiB limit")
        safe_name = sanitize_filename(filename, default_stem="input")
        dest = safe_resolve_under(self.inputs_dir, safe_name)
        if dest.exists() and not overwrite:
            stem, ext = os.path.splitext(safe_name)
            safe_name = f"{stem}_{secrets.token_hex(3)}{ext}"
            dest = safe_resolve_under(self.inputs_dir, safe_name)
        dest.write_bytes(data)
        return {"name": safe_name, "subfolder": "", "type": "input"}

    def resolve_view_file(self, filename: str, file_type: str = "output", subfolder: str = "") -> Path:
        base = self.inputs_dir if file_type == "input" else self.outputs_dir
        target = safe_resolve_under(base, filename, subfolder=subfolder)
        if not target.is_file():
            raise BatchError(404, "not_found", f"File {filename!r} not found")
        return target

    def enqueue(
        self,
        prompt: dict[str, Any],
        submitted_by: str,
        title: str = "",
        preset: str = "",
        timeout_s: int = 300,
        allow_booting: bool = False,
    ) -> BatchJob:
        if not isinstance(prompt, dict) or not prompt:
            raise BatchError(400, "invalid_prompt", "prompt must be a non-empty ComfyUI API graph dictionary")
        for nid, node in prompt.items():
            if not isinstance(node, dict) or not isinstance(node.get("class_type"), str):
                raise BatchError(400, "invalid_prompt", f"Node {nid!r} is missing 'class_type' (API format required)")

        st = self.session.machine.state
        if st != State.READY and not (allow_booting and self.session.machine.is_booting):
            raise BatchError(503, "asleep", f"ComfyUI batch runtime is {st.value}; POST /_gate/wake first")

        self.start()
        self._seq += 1
        prompt_id = f"job_{int(time.time())}_{secrets.token_hex(4)}"
        job = BatchJob(
            prompt_id=prompt_id,
            number=self._seq,
            prompt=prompt,
            submitted_by=submitted_by,
            created_at=time.time(),
            title=(title or "")[:120],
            preset=(preset or "")[:80],
            timeout_s=max(15, min(1800, int(timeout_s or 300))),
            status="waiting_gpu" if st != State.READY else "queued",
        )
        self._jobs[prompt_id] = job
        self._queue.put_nowait(job)
        self.session.touch()
        self._sync_busy_flag()
        self._persist_jobs()
        self.session.bus.publish("job", job.to_summary())
        return job

    def get_job(self, prompt_id: str) -> BatchJob | None:
        return self._jobs.get(prompt_id)

    def list_jobs(self, limit: int = 40) -> list[dict[str, Any]]:
        ordered = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[: max(1, min(100, limit))]
        return [j.to_summary() for j in ordered]

    def history_dict(self, prompt_id: str | None = None) -> dict[str, Any]:
        if prompt_id is not None:
            j = self._jobs.get(prompt_id)
            if not j or j.status not in ("success", "error"):
                return {}
            return {prompt_id: j.to_comfy_history_entry()}
        return {
            pid: j.to_comfy_history_entry()
            for pid, j in self._jobs.items()
            if j.status in ("success", "error")
        }

    def queue_dict(self) -> dict[str, Any]:
        running = [[self._running.number, self._running.prompt_id]] if self._running else []
        pending = [
            [j.number, j.prompt_id]
            for j in sorted(self._jobs.values(), key=lambda x: x.number)
            if j.status in ("queued", "waiting_gpu")
        ]
        return {"queue_running": running, "queue_pending": pending}

    def clear_pending(self) -> int:
        cleared = 0
        kept: list[BatchJob] = []
        while not self._queue.empty():
            j = self._queue.get_nowait()
            j.status = "cancelled"
            j.error = "Cancelled from queue"
            j.finished_at = time.time()
            j.done_event.set()
            self.session.bus.publish("job", j.to_summary())
            cleared += 1
        for j in kept:
            self._queue.put_nowait(j)
        self._sync_busy_flag()
        self._persist_jobs()
        return cleared

    def prune_outputs(self, now: float | None = None) -> int:
        """Delete expired or excess generated files so VPS disk stays bounded."""
        now_ts = now if now is not None else time.time()
        cutoff = now_ts - self.settings.output_retention_days * 86400
        files = [
            p
            for p in self.outputs_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        ]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        removed: set[str] = set()
        for idx, p in enumerate(files):
            try:
                if idx >= self.settings.max_output_files or p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    removed.add(p.name)
            except OSError:
                pass
        if removed:
            for j in self._jobs.values():
                for node_out in j.outputs.values():
                    if "images" in node_out:
                        node_out["images"] = [
                            img for img in node_out["images"] if img.get("filename") not in removed
                        ]
        return len(removed)

    async def _wait_until_session_ready(self) -> bool:
        deadline = time.time() + self.settings.auto_stop.boot_timeout_min * 60
        while time.time() < deadline:
            st = self.session.machine.state
            if st == State.READY:
                return True
            if st in (State.SLEEPING, State.FAILED, State.STOPPING):
                return False
            await asyncio.sleep(0.1)
        return False

    async def _upload_referenced_inputs(self, session_name: str, prompt: dict[str, Any]) -> None:
        if self.session.machine.session_started != self._last_session_started:
            self._uploaded_inputs.clear()
            self._last_session_started = self.session.machine.session_started

        leaves: set[str] = set()
        _collect_string_leaves(prompt, leaves)
        for val in sorted(leaves):
            if "/" in val or "\\" in val or val.startswith("."):
                continue
            candidate = self.inputs_dir / val
            if candidate.is_file():
                mtime = candidate.stat().st_mtime_ns
                if self._uploaded_inputs.get(val) == mtime:
                    continue
                await self.session.colab.upload(session_name, str(candidate), f"/content/ComfyUI/input/{val}")
                self._uploaded_inputs[val] = mtime

    async def _worker_loop(self) -> None:
        while True:
            job = await self._queue.get()
            if job.status == "cancelled":
                self._queue.task_done()
                self._sync_busy_flag()
                continue
            self._running = job
            self._sync_busy_flag()
            try:
                await self._execute_job(job)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.exception("unexpected worker error on %s", job.prompt_id)
                job.status = "error"
                job.error = str(e)[:400]
                job.finished_at = time.time()
            finally:
                job.done_event.set()
                self._running = None
                self._queue.task_done()
                self.session.touch()  # Reset 10-min keep-warm countdown on completion
                self._sync_busy_flag()
                self._persist_jobs()
                self.session.bus.publish("job", job.to_summary())

    async def _execute_job(self, job: BatchJob) -> None:
        if self.session.machine.state != State.READY:
            job.status = "waiting_gpu"
            self.session.bus.publish("job", job.to_summary())
            if not await self._wait_until_session_ready():
                job.status = "error"
                job.error = f"Colab session is not ready ({self.session.machine.state.value})"
                job.finished_at = time.time()
                return

        job.status = "running"
        job.started_at = time.time()
        self.session.touch()
        self.session.bus.publish("job", job.to_summary())

        session_name = self.settings.colab_session_name
        await self._upload_referenced_inputs(session_name, job.prompt)

        remote_task = f"/content/kit/tasks/{job.prompt_id}.json"
        fd = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        try:
            os.chmod(fd.name, 0o600)
            json.dump({"prompt_id": job.prompt_id, "prompt": job.prompt}, fd, ensure_ascii=False)
            fd.close()
            await self.session.colab.upload(session_name, fd.name, remote_task)
        finally:
            Path(fd.name).unlink(missing_ok=True)

        runner_code = (
            "import os, sys\n"
            "if '/content/kit/colab' not in sys.path: sys.path.insert(0, '/content/kit/colab')\n"
            "import headless_runner\n"
            f"try:\n    headless_runner.run_task_file({remote_task!r})\n"
            f"finally:\n"
            f"    try: os.remove({remote_task!r})\n"
            f"    except OSError: pass\n"
        )

        result_payload: dict[str, Any] | None = None
        err_payload: dict[str, Any] | None = None
        try:
            async for raw_line in self.session.colab.exec_stream(session_name, runner_code, job.timeout_s):
                clean = redact_secrets(raw_line)
                if clean.startswith("KIT_JOB_PROGRESS "):
                    try:
                        prog = json.loads(clean.split(" ", 1)[1])
                        job.step = prog.get("step")
                        job.max_steps = prog.get("max")
                        job.current_node = prog.get("node")
                        self.session.bus.publish("job", job.to_summary())
                    except ValueError:
                        pass
                elif clean.startswith("KIT_JOB_RESULT "):
                    result_payload = json.loads(clean.split(" ", 1)[1])
                elif clean.startswith("KIT_JOB_ERROR "):
                    err_payload = json.loads(clean.split(" ", 1)[1])
                else:
                    self.session.bus.publish("log", f"[{job.prompt_id}] {clean}")
        except ColabError as ce:
            if err_payload is None:
                err_payload = {"error": str(ce)[:400], "oom": "out of memory" in str(ce).lower()}

        if err_payload is not None or result_payload is None:
            err_msg = (err_payload or {}).get("error") or "Headless execution returned no KIT_JOB_RESULT"
            if (err_payload or {}).get("oom"):
                self.session.bus.publish("log", f"[{job.prompt_id}] CUDA OOM detected; restarting Colab kernel to reclaim VRAM")
                try:
                    await self.session.colab.restart_kernel(session_name)
                except ColabError as rke:
                    log.warning("restart_kernel after OOM failed: %s", rke)
            job.status = "error"
            job.error = str(err_msg)[:500]
            job.finished_at = time.time()
            return

        # Download output artifacts to local VPS outputs_dir
        local_outputs: dict[str, Any] = {}
        raw_outputs = result_payload.get("outputs") or {}
        for node_id, node_out in raw_outputs.items():
            downloaded_imgs = []
            for img in (node_out or {}).get("images") or []:
                orig_name = str(img.get("filename") or "output.png")
                remote_path = str(img.get("remote_path") or "")
                if not remote_path:
                    continue
                safe_base = sanitize_filename(orig_name, default_stem="output")
                local_name = f"{job.prompt_id}_{safe_base}"
                local_dest = safe_resolve_under(self.outputs_dir, local_name)
                await self.session.colab.download(session_name, remote_path, str(local_dest))
                if local_dest.is_file():
                    downloaded_imgs.append(
                        {
                            "filename": local_name,
                            "subfolder": "",
                            "type": "output",
                        }
                    )
            if downloaded_imgs:
                local_outputs[str(node_id)] = {"images": downloaded_imgs}

        job.outputs = local_outputs
        job.duration_s = result_payload.get("duration_s") or round(time.time() - (job.started_at or job.created_at), 2)
        job.finished_at = time.time()
        job.status = "success"
        self.prune_outputs()
