"""Exercise cancellation against real local child processes, never Google."""
import asyncio
import sys

import pytest
import pytest_asyncio

from gate.colab import ColabCLI, ColabConnectionError, ColabError


@pytest_asyncio.fixture
async def local_cli(monkeypatch):
    original = asyncio.create_subprocess_exec
    children = []
    started = asyncio.Event()

    async def spawn(*argv, **kwargs):
        proc = await original(
            sys.executable, "-u", "-c",
            "import time; print('LOCAL_READY', flush=True); time.sleep(60)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        children.append(proc)
        started.set()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    try:
        yield ColabCLI("unused-local-test"), children, started
    finally:
        for proc in children:
            if proc.returncode is None:
                proc.kill()
            await asyncio.wait_for(proc.wait(), timeout=5)


async def test_command_timeout_reaps_cli_child(local_cli):
    cli, children, _ = local_cli
    with pytest.raises(ColabError, match="timed out"):
        await cli._run("status", timeout=0.1)
    assert children[0].returncode is not None


async def test_command_cancellation_reaps_cli_child(local_cli):
    cli, children, started = local_cli
    task = asyncio.create_task(cli._run("status"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert children[0].returncode is not None


async def test_stream_local_deadline_reaps_cli_child(local_cli, monkeypatch):
    cli, children, _ = local_cli
    original_wait = asyncio.wait_for

    async def bounded_wait(awaitable, timeout):
        return await original_wait(awaitable, min(timeout, 0.05))

    monkeypatch.setattr(asyncio, "wait_for", bounded_wait)
    with pytest.raises(ColabConnectionError, match="timed out; execution completion unknown"):
        async with asyncio.timeout(1):
            async for _ in cli.exec_stream("owned-session", "pass", 30):
                pass
    assert children[0].returncode is not None


async def test_stream_closure_reaps_cli_child(local_cli):
    cli, children, _ = local_cli
    stream = cli.exec_stream("owned-session", "pass", 30)
    assert await anext(stream) == "LOCAL_READY"
    await stream.aclose()
    assert children[0].returncode is not None


@pytest.mark.parametrize("fatal", [
    "RuntimeError: Connection was lost.",
    "colab_cli.runtime.ExecutionError: Execution did not complete",
])
async def test_fatal_transport_traceback_reaps_hung_cli_with_original_error(monkeypatch, fatal):
    original = asyncio.create_subprocess_exec
    children = []

    async def spawn(*argv, **kwargs):
        proc = await original(
            sys.executable, "-u", "-c",
            f"import time; print({fatal!r}, flush=True); time.sleep(60)",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        children.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    try:
        with pytest.raises(ColabError, match="Connection was lost|Execution did not complete"):
            async with asyncio.timeout(1):
                async for _ in ColabCLI("local-test").exec_stream("owned-session", "pass", 30):
                    pass
        assert children[0].returncode is not None
    finally:
        for proc in children:
            if proc.returncode is None:
                proc.kill()
            await asyncio.wait_for(proc.wait(), 2)
