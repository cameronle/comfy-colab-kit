"""Transport/stop regressions. No Colab allocation or external connections."""
import asyncio
from pathlib import Path

import pytest

from aiohttp.test_utils import TestClient, TestServer

from gate.app import BATCH, SESSION, build_app
from gate.config import Settings
from gate.profiles import load_profile
from gate.state import State
from tests.fakes import FakeColab

ROOT = Path(__file__).resolve().parent.parent


def make_app(tmp_path, colab):
    settings = Settings.from_env({
        "AUTH_DISABLED": "true", "STATE_DIR": str(tmp_path), "DEFAULT_PROFILE": "tiny",
    })
    profiles = {"tiny": load_profile(ROOT / "tests/fixtures/profiles/tiny")}
    return build_app(settings, colab=colab, profiles=profiles, run_background=False)


async def test_ready_marker_does_not_allow_dispatch_before_boot_connection_closes(tmp_path):
    class SlowClose(FakeColab):
        def __init__(self):
            super().__init__()
            self.ready_emitted = asyncio.Event()
            self.allow_close = asyncio.Event()
            self.boot_connected = False
            self.overlapped = False

        async def exec_stream(self, name, code, timeout_s):
            if "run_task_file" in code:
                self.overlapped |= self.boot_connected
                async for line in super().exec_stream(name, code, timeout_s):
                    yield line
                return
            self.boot_connected = True
            try:
                async for line in super().exec_stream(name, code, timeout_s):
                    yield line
                self.ready_emitted.set()
                await self.allow_close.wait()
            finally:
                self.boot_connected = False

        async def ensure_dirs(self, name, paths):
            self.overlapped |= self.boot_connected
            await super().ensure_dirs(name, paths)

    remote = SlowClose()
    app = make_app(tmp_path, remote)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post("/prompt", json={
            "prompt": {"1": {"class_type": "SaveImage", "inputs": {}}}, "auto_wake": True,
        })
        job = app[BATCH].get_job((await response.json())["prompt_id"])
        await asyncio.wait_for(remote.ready_emitted.wait(), 2)
        await asyncio.sleep(0.2)
        assert app[SESSION].machine.state != State.READY
        assert not remote.overlapped
        remote.allow_close.set()
        await asyncio.wait_for(job.done_event.wait(), 2)
        assert job.status == "success", job.error
        assert not remote.overlapped
    finally:
        remote.allow_close.set()
        await client.close()


async def test_stop_cancels_running_exec_and_pending_jobs_before_returning(tmp_path):
    class HungJob(FakeColab):
        def __init__(self):
            super().__init__()
            self.running = asyncio.Event()
            self.reaped = asyncio.Event()

        async def exec_stream(self, name, code, timeout_s):
            if "run_task_file" not in code:
                async for line in super().exec_stream(name, code, timeout_s):
                    yield line
                return
            self.running.set()
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                self.reaped.set()

    remote = HungJob()
    app = make_app(tmp_path, remote)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        body = {"prompt": {"1": {"class_type": "SaveImage", "inputs": {}}}, "auto_wake": True}
        response = await client.post("/prompt", json=body)
        first = app[BATCH].get_job((await response.json())["prompt_id"])
        await asyncio.wait_for(remote.running.wait(), 2)
        response = await client.post("/prompt", json=body)
        second = app[BATCH].get_job((await response.json())["prompt_id"])
        response = await client.post("/_gate/stop", json={})
        assert response.status == 200
        assert remote.reaped.is_set()
        assert first.status == second.status == "cancelled"
        assert first.done_event.is_set() and second.done_event.is_set()
        assert not app[SESSION].queue_busy
        assert app[BATCH].queue_dict() == {"queue_running": [], "queue_pending": []}
        assert not remote.live
    finally:
        await client.close()


@pytest.mark.parametrize("failure", [
    "RuntimeError: Connection was lost.",
    "colab exec timed out; execution completion unknown",
])
async def test_transport_failure_releases_runtime_without_replaying_queued_job(tmp_path, failure):
    from gate.colab import ColabConnectionError

    class LostConnection(FakeColab):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.fail = asyncio.Event()

        async def exec_stream(self, name, code, timeout_s):
            if "run_task_file" in code:
                self.started.set()
                await self.fail.wait()
                raise ColabConnectionError(failure)
            async for line in super().exec_stream(name, code, timeout_s):
                yield line

    remote = LostConnection()
    app = make_app(tmp_path, remote)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        body = {"prompt": {"1": {"class_type": "SaveImage", "inputs": {}}}, "auto_wake": True}
        response = await client.post("/prompt", json=body)
        first = app[BATCH].get_job((await response.json())["prompt_id"])
        await asyncio.wait_for(remote.started.wait(), 2)
        response = await client.post("/prompt", json=body)
        second = app[BATCH].get_job((await response.json())["prompt_id"])
        remote.fail.set()
        await asyncio.wait_for(first.done_event.wait(), 2)
        assert first.status == "error" and failure in first.error
        assert not remote.live
        assert second.status == "cancelled"
        assert not app[SESSION].queue_busy
        assert app[SESSION].machine.state == State.SLEEPING
        assert sum(c.startswith("new:") for c in remote.calls) == 1
    finally:
        await client.close()
