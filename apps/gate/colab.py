"""Thin async wrapper around the official `colab` CLI (google-colab-cli).

The gate is the only component holding the Colab OAuth token (COLAB_TOKEN_DIR).
Management CLI invocations explicitly pass `--auth oauth2`; the execution
adapter fixes AuthProvider.OAUTH2 directly, never ambient ADC or service accounts.
Secrets for the VM are sent on stdin (never on argv/process listings).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import AsyncIterator, Protocol

from .redaction import redact_secrets

log = logging.getLogger("gate.colab")
EXEC_GRACE_S = 15


class ColabError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(redact_secrets(message))


class ColabConnectionError(ColabError):
    """Execution completion is unknown; never automatically replay the task."""


class ColabRunner(Protocol):
    async def sessions(self) -> list[str]: ...
    async def new(self, name: str, accelerator: str) -> None: ...
    async def ensure_dirs(self, name: str, paths: list[str]) -> None: ...
    async def upload(self, name: str, src: str, dst: str) -> None: ...
    async def download(self, name: str, remote_path: str, local_path: str) -> None: ...
    def exec_stream(self, name: str, code: str, timeout_s: int) -> AsyncIterator[str]: ...
    async def restart_kernel(self, name: str) -> None: ...
    async def check_ready(self, name: str) -> bool: ...
    async def stop(self, name: str) -> None: ...
    async def ccu_info(self) -> float | None: ...


class ColabCLI:
    def __init__(self, binary: str = "colab", token_dir: str = "", *,
                 exec_python: str = "", connect_timeout_s: float = 45):
        self.binary = binary
        self.exec_python = exec_python or sys.executable
        self.connect_timeout_s = connect_timeout_s
        self.env = dict(os.environ)
        if token_dir:
            # google-colab-cli keeps its token under ~/.config/colab-cli; point HOME-relative config there
            self.env["XDG_CONFIG_HOME"] = token_dir
        # the CLI is Python: without this its stdout is block-buffered on a pipe and `exec` output
        # (setup/job progress) would only arrive in 4-8 KB chunks or at the very end
        self.env["PYTHONUNBUFFERED"] = "1"

    async def _run(self, *args: str, stdin: str | None = None, timeout: float = 600) -> str:
        proc = await asyncio.create_subprocess_exec(
            self.binary,
            "--auth",
            "oauth2",
            *args,
            env=self.env,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(stdin.encode() if stdin else None), timeout)
        except asyncio.TimeoutError as e:
            raise ColabError(f"colab {args[0]} timed out") from e
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()
        text = out.decode(errors="replace")
        if proc.returncode:
            safe_text = redact_secrets(text)
            raise ColabError(f"colab {args[0]} failed ({proc.returncode}): {safe_text[-500:]}")
        return text

    async def sessions(self) -> list[str]:
        try:
            out = await self._run("sessions", timeout=60)
        except ColabError as e:
            msg = str(e).lower()
            if "no active session" in msg or "no session" in msg:
                return []
            raise
        return parse_sessions(out)

    async def new(self, name: str, accelerator: str) -> None:
        await self._run("new", "-s", name, "--gpu", accelerator, timeout=600)

    async def upload(self, name: str, src: str, dst: str) -> None:
        await self._run("upload", "-s", name, src, dst, timeout=600)

    async def ensure_dirs(self, name: str, paths: list[str]) -> None:
        # Contents API uploads do not create parent directories. Send paths on
        # stdin, safely quoted as Python literals, before uploading any task.
        code = (
            f"import os\nfor _kit_dir in {paths!r}:\n    os.makedirs(_kit_dir, exist_ok=True)\n"
            "print('KIT_DIRS_READY', flush=True)\n"
        )
        for attempt in range(2):
            try:
                # Stream even this small command: CLI connection tracebacks can
                # leave its process alive, hiding the real cause in communicate().
                confirmed, tail = False, ""
                async for line in self.exec_stream(name, code, 20):
                    if line == "KIT_DIRS_READY":
                        confirmed = True
                    else:
                        tail = (tail + line + " ")[-400:]
                # The official CLI can exit zero after a remote Python error.
                # Wait for connection closure AND an explicit success marker.
                if not confirmed:
                    raise ColabError(
                        "Colab directory preparation failed before task upload: "
                        f"missing KIT_DIRS_READY confirmation; {tail.strip() or 'no CLI output'}"
                    )
                return
            except ColabConnectionError as e:
                if attempt:
                    raise ColabConnectionError(
                        "Colab directory preparation failed before task upload after 2 attempts: "
                        f"{e}"
                    ) from e
                # Only mkdir is replayed. No graph has been uploaded or run.
                log.warning("directory preparation connection failed; retrying idempotent mkdir once: %s", e)
                await asyncio.sleep(1)

    async def download(self, name: str, remote_path: str, local_path: str) -> None:
        await self._run("download", "-s", name, remote_path, local_path, timeout=600)

    async def exec_stream(self, name: str, code: str, timeout_s: int):
        proc = await asyncio.create_subprocess_exec(
            self.exec_python,
            str(Path(__file__).with_name("colab_exec.py")),
            "--session",
            name,
            "--timeout",
            str(timeout_s),
            "--connect-timeout",
            str(self.connect_timeout_s),
            env=self.env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            assert proc.stdin and proc.stdout
            # Setup/auth/handshake must not consume the task's execution budget.
            # Each phase is bounded; duplicate markers cannot extend a deadline.
            connected = False
            deadline = asyncio.get_running_loop().time() + self.connect_timeout_s + EXEC_GRACE_S
            proc.stdin.write(code.encode())
            await asyncio.wait_for(proc.stdin.drain(), max(0.0, deadline - asyncio.get_running_loop().time()))
            proc.stdin.close()
            while True:
                remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                raw = await asyncio.wait_for(proc.stdout.readline(), remaining)
                if not raw:
                    break
                line = redact_secrets(raw.decode(errors="replace").rstrip("\n"))
                if line == "KIT_COLAB_EXEC_READY":
                    if not connected:
                        connected = True
                        deadline = asyncio.get_running_loop().time() + timeout_s + EXEC_GRACE_S
                    continue
                if "KIT_COLAB_CODE_ERROR:" in line:
                    raise ColabError(f"Colab remote code failed: {line}")
                # CLI 0.7.5 can leave non-daemon Jupyter threads alive when
                # its preliminary exec fails before entering runtime.stop().
                # Preserve the cause and reap the owned child immediately.
                if any(marker in line for marker in (
                    "KIT_COLAB_CONNECTION_ERROR:",
                    "RuntimeError: Connection was lost.",
                    "ExecutionError: Execution did not complete",
                    "WebSocketConnectionClosedException:",
                )):
                    raise ColabConnectionError(f"Colab kernel connection failed: {line}")
                yield line
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            rc = await asyncio.wait_for(proc.wait(), remaining)
            if rc:
                raise ColabError(f"colab exec exited with {rc}")
            if not connected:
                raise ColabConnectionError("Colab execution readiness confirmation missing; task not confirmed")
        except asyncio.TimeoutError as e:
            # A local deadline does not cancel remote notebook execution.
            # The dispatcher must fail closed, never start a queued graph on it.
            raise ColabConnectionError("colab exec timed out; execution completion unknown") from e
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()

    async def restart_kernel(self, name: str) -> None:
        await self._run("restart-kernel", "-s", name, timeout=120)

    async def check_ready(self, name: str) -> bool:
        try:
            code = "import os; print('KIT_OK' if os.path.isfile('/content/kit/runtime_config.json') else 'KIT_MISSING')\n"
            confirmed = False
            async for line in self.exec_stream(name, code, 15):
                confirmed |= line == "KIT_OK"
            return confirmed
        except Exception as e:
            log.warning("check_ready via colab exec failed: %s", e)
            return False

    async def stop(self, name: str) -> None:
        try:
            await self._run("stop", "-s", name, timeout=180)
        except ColabError as e:
            msg = str(e).lower()
            if any(k in msg for k in ("not found", "no such", "404", "unknown")):
                log.info("colab stop %s: instance already released (%s)", name, e)
                return
            raise

    async def ccu_info(self) -> float | None:
        try:
            out = await self._run("usage", timeout=60)
        except ColabError:
            return None
        return parse_ccu_balance(out)


def parse_sessions(text: str) -> list[str]:
    """`colab sessions` (0.7.x) prints one line per server-side assignment:
    `[name] endpoint | Hardware: L4 | Shape: ... | Variant: GPU`; `[?]` = assignment with no local name."""
    return [
        m.group(1)
        for line in text.splitlines()
        if " | " in line and (m := re.match(r"^\s*\[([^\]]+)\]\s", line))
    ]


def parse_ccu_balance(text: str) -> float | None:
    """Best effort: accept JSON with a balance-like key or a 'balance: 123.4' line."""
    try:
        d = json.loads(text)
        for k in ("currentBalance", "balance", "remaining", "ccu"):
            if k in d:
                return float(d[k])
    except (ValueError, TypeError):
        pass
    m = re.search(r"(balance|remaining)[^0-9]*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    return float(m.group(2)) if m else None
