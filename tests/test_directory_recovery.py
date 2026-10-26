"""Directory-setup recovery using owned local children, never a Colab VM."""
import asyncio
import sys

import pytest
import pytest_asyncio

from gate.colab import ColabCLI, ColabConnectionError, ColabError


@pytest_asyncio.fixture
async def local_commands(monkeypatch):
    spawn_real = asyncio.create_subprocess_exec
    sleep_real = asyncio.sleep
    children = []
    attempts = []
    actions = []

    async def spawn(*argv, **kwargs):
        index = len(attempts)
        attempts.append(argv)
        action = actions[min(index, len(actions) - 1)]
        if action == "disconnect":
            script = (
                "import time; "
                "print('RuntimeError: Connection was lost.', flush=True); "
                "time.sleep(60)"
            )
        elif action == "hang":
            script = "import time; time.sleep(60)"
        elif action == "unconfirmed":
            script = "print('PermissionError: cannot create remote directory', flush=True)"
        else:
            script = "import sys; exec(compile(sys.stdin.read(), '<local-mkdir>', 'exec'))"
        proc = await spawn_real(
            sys.executable, "-u", "-c", script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        children.append(proc)
        return proc

    async def no_backoff(seconds):
        await sleep_real(0 if seconds == 1 else seconds)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(asyncio, "sleep", no_backoff)
    try:
        yield ColabCLI("local-only"), actions, children, attempts
    finally:
        for proc in children:
            if proc.returncode is None:
                proc.kill()
            await asyncio.wait_for(proc.wait(), 2)


async def test_directory_preparation_retries_a_lost_connection_without_running_a_graph(local_commands, tmp_path):
    cli, actions, children, attempts = local_commands
    actions.extend(["disconnect", "mkdir"])
    target = tmp_path / "odd'parent/tasks"
    async with asyncio.timeout(3):
        await cli.ensure_dirs("owned-session", [str(target)])
    assert target.is_dir()
    assert len(attempts) == 2
    assert all(proc.returncode is not None for proc in children)


async def test_directory_preparation_retries_a_silent_connection_timeout(local_commands, tmp_path, monkeypatch):
    cli, actions, children, attempts = local_commands
    actions.extend(["hang", "mkdir"])
    wait_real = asyncio.wait_for

    async def short_deadline(awaitable, timeout):
        return await wait_real(awaitable, min(timeout, 0.25))

    monkeypatch.setattr(asyncio, "wait_for", short_deadline)
    target = tmp_path / "tasks"
    async with asyncio.timeout(3):
        await cli.ensure_dirs("owned-session", [str(target)])
    assert target.is_dir()
    assert len(attempts) == 2
    assert all(proc.returncode is not None for proc in children)


async def test_directory_preparation_requires_explicit_confirmation_without_retrying_code_errors(local_commands, tmp_path):
    cli, actions, children, attempts = local_commands
    actions.append("unconfirmed")
    with pytest.raises(ColabError, match="directory preparation.*before task upload.*PermissionError"):
        async with asyncio.timeout(3):
            await cli.ensure_dirs("owned-session", [str(tmp_path / "tasks")])
    assert len(attempts) == 1
    assert all(proc.returncode is not None for proc in children)


async def test_directory_preparation_exhausts_exactly_two_attempts_with_stage_error(local_commands, tmp_path):
    cli, actions, children, attempts = local_commands
    actions.append("disconnect")
    with pytest.raises(ColabConnectionError, match="directory preparation failed before task upload after 2 attempts"):
        async with asyncio.timeout(3):
            await cli.ensure_dirs("owned-session", [str(tmp_path / "tasks")])
    assert len(attempts) == 2
    assert all(proc.returncode is not None for proc in children)
