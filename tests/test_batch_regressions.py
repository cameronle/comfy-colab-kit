"""Cold-start regressions; all remote files and executors stay local to pytest."""
import asyncio
import json
import pytest
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from gate.app import BATCH, build_app
from gate.colab import ColabError
from gate.config import Settings
from gate.profiles import load_profile
from tests.fakes import FAKE_PNG, FakeColab

ROOT = Path(__file__).resolve().parent.parent


class StrictTaskColab(FakeColab):
    """Model Contents API semantics: PUT cannot create missing parent dirs."""

    def __init__(self, root):
        super().__init__()
        self.root = root
        (root / "content/kit").mkdir(parents=True)

    async def ensure_dirs(self, name, paths):
        self.calls.append("ensure_dirs")
        for path in paths:
            (self.root / path.lstrip("/")).mkdir(parents=True, exist_ok=True)

    async def upload(self, name, src, dst):
        if dst.startswith("/content/kit/tasks/"):
            parent = (self.root / dst.lstrip("/")).parent
            if not parent.is_dir():
                raise ColabError("Contents API HTTP 500: parent directory does not exist")
        await super().upload(name, src, dst)


async def test_directory_preparation_executes_idempotent_safely_quoted_mkdir(tmp_path, monkeypatch):
    import subprocess
    import sys
    from gate.colab import ColabCLI

    paths = [str(tmp_path / "nested/tasks"), str(tmp_path / "odd'directory")]
    cli = ColabCLI("unused-local-test")

    async def run(*args, stdin=None, timeout=None):
        assert args == ("exec", "-s", "owned-session", "--timeout", "20")
        assert timeout == 30
        subprocess.run([sys.executable, "-c", stdin], check=True, timeout=5)
        return ""

    monkeypatch.setattr(cli, "_run", run)
    await cli.ensure_dirs("owned-session", paths)
    await cli.ensure_dirs("owned-session", paths)
    assert all(Path(p).is_dir() for p in paths)


def test_log_redacts_runtime_proxy_query_parameters():
    from gate.session import redact_secrets

    secret = "runtime-test-value-" + "x" * 32
    error = f"500 Server Error for url: https://runtime.invalid/api/contents/task.json?authuser=0&colab-runtime-proxy-token={secret}&content=1"
    clean = redact_secrets(error)
    assert secret not in clean
    assert "500 Server Error" in clean
    assert "authuser=0" in clean
    assert "content=1" in clean


def test_restored_session_logs_and_reason_are_redacted(tmp_path):
    from gate.session import Session

    secret = "runtime-test-value-" + "x" * 32
    raw_error = f"HTTP 500 ?colab-runtime-proxy-token={secret}"
    (tmp_path / "session.log").write_text(raw_error + "\n")
    (tmp_path / "session.json").write_text(json.dumps({"state": "sleeping", "reason": raw_error}))
    session = Session(Settings.from_env({"AUTH_DISABLED": "true", "STATE_DIR": str(tmp_path)}), {}, FakeColab())
    session.restore()
    assert all(secret not in line for line in session.bus.logs)
    assert secret not in session.status()["reason"]


async def test_batch_upload_errors_are_redacted_in_api_history_disk_and_logs(tmp_path, caplog):
    from gate.app import SESSION

    secret = "runtime-test-value-" + "x" * 32

    class UploadFailure(FakeColab):
        async def upload(self, name, src, dst):
            if "/tasks/" in dst:
                raise ColabError(f"HTTP 500 https://runtime.invalid/task?colab-runtime-proxy-token={secret}")
            await super().upload(name, src, dst)

    settings = Settings.from_env({"AUTH_DISABLED": "true", "STATE_DIR": str(tmp_path), "DEFAULT_PROFILE": "tiny"})
    profiles = {"tiny": load_profile(ROOT / "tests/fixtures/profiles/tiny")}
    app = build_app(settings, colab=UploadFailure(), profiles=profiles, run_background=False)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post("/prompt", json={"prompt": {"1": {"class_type": "SaveImage", "inputs": {}}}, "auto_wake": True})
        assert response.status == 200
        job = app[BATCH].get_job((await response.json())["prompt_id"])
        await asyncio.wait_for(job.done_event.wait(), timeout=5)
        assert job.status == "error"
        assert secret not in job.error
        assert secret not in json.dumps(job.to_summary())
        assert secret not in await (await client.get(f"/history/{job.prompt_id}")).text()
        assert secret not in (tmp_path / "jobs.json").read_text()
        assert secret not in caplog.text
        app[SESSION].bus.publish("log", f"runtime URL ?colab-runtime-proxy-token={secret}")
        assert all(secret not in line for line in app[SESSION].bus.logs)
    finally:
        await client.close()


@pytest.mark.parametrize("failure", ["no_images", "empty_download"])
async def test_batch_never_claims_success_without_nonempty_artifacts(tmp_path, failure):
    class MissingArtifact(FakeColab):
        async def exec_stream(self, name, code, timeout_s):
            async for line in super().exec_stream(name, code, timeout_s):
                if failure == "no_images" and line.startswith("KIT_JOB_RESULT "):
                    payload = json.loads(line.split(" ", 1)[1])
                    payload["outputs"] = {}
                    line = "KIT_JOB_RESULT " + json.dumps(payload)
                yield line

        async def download(self, name, remote_path, local_path):
            Path(local_path).write_bytes(b"")

    settings = Settings.from_env({"AUTH_DISABLED": "true", "STATE_DIR": str(tmp_path), "DEFAULT_PROFILE": "tiny"})
    profiles = {"tiny": load_profile(ROOT / "tests/fixtures/profiles/tiny")}
    app = build_app(settings, colab=MissingArtifact(), profiles=profiles, run_background=False)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post("/prompt", json={"prompt": {"1": {"class_type": "SaveImage", "inputs": {}}}, "auto_wake": True})
        assert response.status == 200
        job = app[BATCH].get_job((await response.json())["prompt_id"])
        await asyncio.wait_for(job.done_event.wait(), timeout=5)
        assert job.status == "error"
        assert "image" in job.error.lower() or "artifact" in job.error.lower()
    finally:
        await client.close()


async def test_batch_creates_missing_remote_task_directory_before_upload(tmp_path):
    settings = Settings.from_env({
        "AUTH_DISABLED": "true",
        "STATE_DIR": str(tmp_path / "state"),
        "DEFAULT_PROFILE": "tiny",
    })
    colab = StrictTaskColab(tmp_path / "remote")
    profiles = {"tiny": load_profile(ROOT / "tests/fixtures/profiles/tiny")}
    app = build_app(settings, colab=colab, profiles=profiles, run_background=False)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        for attempt in range(2):
            response = await client.post("/prompt", json={
                "prompt": {"1": {"class_type": "SaveImage", "inputs": {}}},
                "auto_wake": True,
            })
            assert response.status == 200
            job = app[BATCH].get_job((await response.json())["prompt_id"])
            await asyncio.wait_for(job.done_event.wait(), timeout=5)
            assert job.status == "success", job.error
            image = job.to_summary()["images"][0]
            output = await client.get(image["url"])
            assert output.status == 200
            assert await output.read() == FAKE_PNG
            if attempt == 0:
                # A reused VM can lose this directory; dispatch must repair it.
                (colab.root / "content/kit/tasks").rmdir()
        assert sum(c.startswith("new:") for c in colab.calls) == 1
    finally:
        await client.close()
