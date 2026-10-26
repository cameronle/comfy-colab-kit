"""Real Jupyter client + delayed loopback WebSocket, never Google or a GPU."""
import asyncio
import importlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer


@pytest_asyncio.fixture
async def jupyter_server():
    state = {"delay": 0, "codes": [], "sockets": [], "kernel_gets": 0, "kernel_creates": 0}

    async def kernel(request):
        state["kernel_gets"] += 1
        await asyncio.sleep(state.get("kernel_delay", 0))
        if state.get("missing_kernel"):
            raise web.HTTPNotFound()
        return web.json_response({
            "id": "fixture-kernel", "name": "python3", "execution_state": "idle",
            "last_activity": "2026-01-01T00:00:00Z", "connections": 0,
        })

    async def channels(request):
        await asyncio.sleep(state["delay"])
        ws = web.WebSocketResponse()
        try:
            await ws.prepare(request)
            state["sockets"].append(ws)
            async for frame in ws:
                if frame.type != web.WSMsgType.TEXT:
                    continue
                message = json.loads(frame.data)
                if message["header"]["msg_type"] != "execute_request":
                    continue
                code = message["content"]["code"]
                state["codes"].append(code)
                if "LOCAL_NO_REPLY" in code:
                    continue
                failed = state.get("error_at") == len(state["codes"])
                reply_content = ({"status": "error", "ename": "ValueError", "evalue": "fixture code error"}
                                 if failed else {"status": "ok", "execution_count": 1,
                                                 "user_expressions": {}, "payload": []})
                for kind, channel, content in [
                    ("stream", "iopub", {"name": "stdout", "text": "LOCAL_FIXTURE_ACK\n"}),
                    ("execute_reply", "shell", reply_content),
                    ("status", "iopub", {"execution_state": "idle"}),
                ]:
                    header = {"msg_id": uuid.uuid4().hex, "msg_type": kind,
                              "session": "fixture-session", "username": "fixture",
                              "date": datetime.now(timezone.utc).isoformat(), "version": "5.4"}
                    await ws.send_json({
                        "header": header, "msg_id": header["msg_id"], "msg_type": kind,
                        "parent_header": message["header"], "metadata": {},
                        "channel": channel, "content": content,
                    })
        except (ConnectionResetError, RuntimeError):
            # A rejected/delayed handshake may have already been cancelled.
            pass
        finally:
            await ws.close()
        return ws

    async def create_kernel(request):
        state["kernel_creates"] += 1
        assert (await request.json())["name"] == "python3"
        return web.json_response({"id": "fixture-kernel", "name": "python3", "execution_state": "idle"})

    app = web.Application()
    app.router.add_post("/api/kernels", create_kernel)
    app.router.add_get("/api/kernels/fixture-kernel", kernel)
    app.router.add_get("/api/kernels/fixture-kernel/channels", channels)
    server = TestServer(app)
    await server.start_server()
    state["url"] = str(server.make_url("/")).rstrip("/")
    try:
        yield state
    finally:
        for ws in state["sockets"]:
            await ws.close()
        await server.close()


async def test_real_client_waits_past_old_ten_second_handshake_limit(jupyter_server):
    # Deliberately exceed the exact default that failed in production.
    jupyter_server["delay"] = 12
    adapter_path = Path(__file__).resolve().parents[1] / "apps/gate/colab_exec.py"
    assert adapter_path.is_file(), "Missing readiness-gated Colab execution adapter"
    adapter = importlib.import_module("gate.colab_exec")
    code = "  # Complete original prompt: 中文 🍊\npass\n"
    runtime = await asyncio.to_thread(
        adapter.connect_runtime, jupyter_server["url"], "local-fixture-only",
        kernel_id="fixture-kernel", session_id="fixture-session", connect_timeout_s=30,
    )
    ws = runtime._kernel_client._manager.client
    listener = ws.connection_thread
    try:
        assert ws.connection_ready.is_set()
        outputs = await asyncio.to_thread(runtime.execute_code, code, timeout=1)
        assert outputs == [{"output_type": "stream", "name": "stdout", "text": "LOCAL_FIXTURE_ACK\n"}]
        assert jupyter_server["codes"] == [code]
        assert jupyter_server["kernel_gets"] == 1
    finally:
        await asyncio.to_thread(runtime.stop)
    assert not listener.is_alive()


async def run_adapter_fixture(server, tmp_path, code, connect_timeout=3, via_gateway=False,
                              initial_kernel: str | None = "fixture-kernel"):
    import os
    import sys
    from datetime import timedelta
    from colab_cli.state import SessionState, StateStore

    home = tmp_path / "isolated-home"
    state_path = home / ".config/colab-cli/sessions.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    store = StateStore(str(state_path))
    store.add(SessionState(
        name="fixture", token="local-fixture-only", url=server["url"], endpoint="fixture-endpoint",
        token_expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
        kernel_id=initial_kernel, session_id="fixture-session" if initial_kernel else None,
    ))
    adapter = Path(__file__).resolve().parents[1] / "apps/gate/colab_exec.py"
    env = dict(os.environ, HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"))
    if via_gateway:
        from gate.colab import ColabCLI
        cli = ColabCLI("unused", exec_python=sys.executable, connect_timeout_s=connect_timeout)
        cli.env = env
        lines = [line async for line in cli.exec_stream("fixture", code, 1)]
        return 0, "\n".join(lines), store.get("fixture")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(adapter), "--session", "fixture", "--timeout", "1",
        "--connect-timeout", str(connect_timeout), env=env,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    communication = asyncio.create_task(proc.communicate(code.encode()))
    try:
        try:
            output, _ = await asyncio.wait_for(asyncio.shield(communication), 10)
        except asyncio.TimeoutError:
            proc.kill()
            output, _ = await communication
            raise AssertionError(f"Local executor did not exit: {output.decode()}")
        return proc.returncode, output.decode(), store.get("fixture")
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


async def test_isolated_executor_runs_prelude_and_unmodified_code_once(jupyter_server, tmp_path):
    jupyter_server["delay"] = 1.5  # Longer than the separate 1-second execution limit.
    code = "  # 中文提示词，不得截断 🍊\nprint('hello')\n"
    rc, output, stored = await run_adapter_fixture(jupyter_server, tmp_path, code)
    assert rc == 0, output
    assert output.count("KIT_COLAB_EXEC_READY") == 1
    assert len(jupyter_server["codes"]) == 2  # Prelude, then exactly one user payload.
    assert jupyter_server["codes"][1] == code
    assert "os.chdir('/content')" in jupyter_server["codes"][0]
    assert stored.running is None
    assert "local-fixture-only" not in output


async def test_silent_execution_times_out_without_spinning_or_replaying(jupyter_server, tmp_path):
    rc, output, stored = await run_adapter_fixture(jupyter_server, tmp_path, "# LOCAL_NO_REPLY\n")
    assert rc == 3, output
    assert "KIT_COLAB_CONNECTION_ERROR: TimeoutError:" in output
    assert len(jupyter_server["codes"]) == 2
    assert stored.running is None


@pytest.mark.parametrize("stage", ["handshake", "kernel_lookup"])
async def test_connection_failure_submits_no_prelude_or_user_code(jupyter_server, tmp_path, stage):
    jupyter_server["delay" if stage == "handshake" else "kernel_delay"] = 0.6
    rc, output, stored = await run_adapter_fixture(jupyter_server, tmp_path, "pass", connect_timeout=0.2)
    assert rc == 3, output
    assert "KIT_COLAB_CONNECTION_ERROR:" in output
    assert "KIT_COLAB_EXEC_READY" not in output
    assert not jupyter_server["codes"]
    assert stored.running is None and stored.kernel_id == "fixture-kernel"
    assert "Traceback" not in output and "local-fixture-only" not in output


@pytest.mark.parametrize("error_at", [1, 2])
async def test_remote_code_error_is_not_a_retriable_transport_failure(jupyter_server, tmp_path, error_at):
    jupyter_server["error_at"] = error_at
    rc, output, stored = await run_adapter_fixture(jupyter_server, tmp_path, "pass")
    assert rc == 2, output
    assert "KIT_COLAB_CODE_ERROR: ValueError: fixture code error" in output
    assert "KIT_COLAB_CONNECTION_ERROR:" not in output
    assert len(jupyter_server["codes"]) == error_at
    assert stored.running is None


async def test_gateway_uses_owned_adapter_end_to_end(jupyter_server, tmp_path):
    code = " # 完整文本 🍊\npass\n"
    rc, output, stored = await run_adapter_fixture(jupyter_server, tmp_path, code, via_gateway=True)
    assert rc == 0 and output == "LOCAL_FIXTURE_ACK"
    assert jupyter_server["codes"][1] == code
    assert stored.running is None


async def test_missing_existing_kernel_does_not_silently_replace_it(jupyter_server, tmp_path):
    jupyter_server["missing_kernel"] = True
    rc, output, stored = await run_adapter_fixture(jupyter_server, tmp_path, "pass")
    assert rc == 3, output
    assert not jupyter_server["codes"]
    assert jupyter_server["kernel_creates"] == 0
    assert stored.kernel_id == "fixture-kernel"


async def test_fresh_runtime_creates_and_persists_its_initial_kernel(jupyter_server, tmp_path):
    rc, output, stored = await run_adapter_fixture(jupyter_server, tmp_path, "pass", initial_kernel=None)
    assert rc == 0, output
    assert jupyter_server["kernel_creates"] == 1
    assert stored.kernel_id == "fixture-kernel" and stored.session_id
    assert len(jupyter_server["codes"]) == 2
