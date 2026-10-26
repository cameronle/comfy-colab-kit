"""Readiness-gated execution using the official Colab auth/runtime libraries.

This module is also an executable script: an isolated Colab CLI interpreter can
run it without importing the gateway's aiohttp stack. Only this owned process
uses the pinned Jupyter client's private manager interface. No installed vendor
files are patched, no VM is allocated here, and user code is never replayed.
"""
from __future__ import annotations

import time


class ConnectionSetupError(RuntimeError):
    """No user code was submitted by this connection attempt."""


def _close_channels(ws):
    import socket

    import threading

    app = ws.kernel_socket
    connection = app.sock if app is not None else None
    raw_socket = connection.sock if connection is not None else None
    if app is not None:
        app.keep_running = False
    if raw_socket is not None:
        try:
            # Wake select, but do not close its fd in this thread: closing the
            # fd first can unregister it from epoll before the listener wakes.
            raw_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    listener = ws.connection_thread
    if listener is not None and listener is not threading.current_thread():
        listener.join(timeout=2)
    ws.timeout = min(ws.timeout, 1)
    ws.stop_channels()


def connect_runtime(url, token, *, kernel_id=None, session_id=None,
                    connect_timeout_s=45, on_kernel_started=None, on_session_started=None):
    """Return an official runtime only after its WebSocket is actually ready.

    REST lookup/kernel startup and WS handshake share one monotonic budget.
    Kernel lifecycle remains the gateway's responsibility, including failures.
    """
    import jupyter_kernel_client
    import websocket
    from colab_cli.runtime import ColabRuntime

    class ConnectedRuntime(ColabRuntime):
        def stop(self, shutdown_kernel=False):
            if self._kernel_client:
                _close_channels(self._kernel_client._manager.client)
                if shutdown_kernel:
                    super().stop(shutdown_kernel=True)

    deadline = time.monotonic() + connect_timeout_s

    def remaining():
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise ConnectionSetupError("Colab connection deadline exceeded before task execution")
        return budget

    from jupyter_kernel_client.manager import KernelHttpManager

    class DeadlineManager(KernelHttpManager):
        def refresh_model(self, timeout=None):
            return super().refresh_model(timeout=remaining())

    class DeadlineClient(jupyter_kernel_client.KernelClient):
        kernel_manager_class = DeadlineManager

        def __init__(self, **kwargs):
            self._manager = None  # Safe destruction if the initial REST lookup fails.
            super().__init__(**kwargs)

    client = None
    ws = None
    try:
        client = DeadlineClient(
            server_url=url, token=token, kernel_id=kernel_id,
            headers={"X-Colab-Client-Agent": "colab-cli", "X-Colab-Runtime-Proxy-Token": token},
            client_kwargs={
                "subprotocol": jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
                "extra_params": {"colab-runtime-proxy-token": token},
            },
        )
        client._own_kernel = False
        if not client.has_kernel:
            if kernel_id:
                raise ConnectionSetupError("Colab kernel no longer exists; task not submitted")
            client._manager.start_kernel(name="python3", timeout=remaining())
        if not kernel_id and on_kernel_started:
            # Persist a newly created kernel before waiting for its transport,
            # so a safe mkdir retry cannot accidentally create another kernel.
            on_kernel_started(client.id)
        ws = client._manager.client
        if session_id:
            ws.session.session = session_id
        elif on_session_started:
            on_session_started(ws.session.session)
        ws.timeout = remaining()
        # The pinned library does not apply this timeout to the underlying
        # handshake socket. Scope its default to this dedicated exec process;
        # otherwise a failed handshake can strand a non-daemon listener thread.
        old_timeout = websocket.getdefaulttimeout()
        try:
            websocket.setdefaulttimeout(ws.timeout)
            ws.start_channels()
        finally:
            websocket.setdefaulttimeout(old_timeout)
        if not ws.connection_ready.is_set():
            raise ConnectionSetupError(
                f"Colab WebSocket not ready within {connect_timeout_s:g}s before task execution"
            )
        runtime = ConnectedRuntime(url, token, kernel_id=client.id, session_id=ws.session.session)
        runtime._kernel_client = client
        runtime._apply_ws_hook()
        return runtime
    except Exception:
        if ws is not None:
            _close_channels(ws)
        # Never call shutdown_kernel or delete a live assignment on a failed
        # transport setup. The parent owns the safe stop/queue-cancel policy.
        raise


def _execute_checked(runtime, code, timeout, *, stream):
    import queue
    from jupyter_kernel_client.client import output_hook

    outputs = []

    def on_output(message):
        changed = output_hook(outputs, message)
        if stream:
            for index in sorted(changed):
                item = outputs[index]
                if item.get("output_type") == "stream":
                    print(_redact(item.get("text", "")), end="", flush=True)
                elif item.get("output_type") == "error":
                    print(_redact(f"{item.get('ename', 'Error')}: {item.get('evalue', '')}"), flush=True)

    ws = runtime.kernel_client._manager.client
    if not ws.connection_ready.is_set():
        raise RuntimeError("Colab WebSocket is not ready before execution")
    deadline = time.monotonic() + timeout
    message_id = ws.execute(code, allow_stdin=False)
    reply, idle = None, False
    # The pinned execute_interactive loop can busy-wait forever after timeout
    # (and races its event clear against incoming frames). Drain real channel
    # queues with bounded blocking reads instead; require both idle and reply.
    while not (idle and reply is not None):
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("Colab execution timed out; execution completion unknown")
        if not ws.connection_ready.is_set():
            raise RuntimeError("Connection was lost; execution completion unknown")
        try:
            message = ws.get_iopub_msg(timeout=min(left, 0.1))
        except queue.Empty:
            pass
        else:
            if message.get("parent_header", {}).get("msg_id") == message_id:
                on_output(message)
                idle |= (message["header"]["msg_type"] == "status"
                         and message["content"].get("execution_state") == "idle")
        while True:
            try:
                candidate = ws.get_shell_msg(timeout=0)
            except queue.Empty:
                break
            if (candidate.get("parent_header", {}).get("msg_id") == message_id
                    and candidate["header"]["msg_type"] == "execute_reply"):
                reply = candidate
    content = reply.get("content", {}) if reply else {}
    if content.get("status") == "error":
        raise RemoteCodeError(_redact(
            f"{content.get('ename', 'Error')}: {content.get('evalue', 'remote code failed')}"
        ))
    if content.get("status") != "ok":
        raise RuntimeError("Execution completion unknown: missing successful kernel reply")
    return outputs


class RemoteCodeError(RuntimeError):
    pass


def _redact(text):
    if __package__:
        from .redaction import redact_secrets
    else:
        from redaction import redact_secrets
    return redact_secrets(text)


def main(argv=None):
    import argparse
    import importlib.metadata
    import math
    import sys
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--connect-timeout", type=float, default=45)
    args = parser.parse_args(argv)
    for value in (args.timeout, args.connect_timeout):
        if not math.isfinite(value) or value <= 0:
            parser.error("timeouts must be positive finite numbers")
    runtime, session, state = None, None, None
    failure = None
    result = 0
    try:
        # This adapter deliberately uses the reviewed 0.9.0 manager contract.
        for package, expected in (("google-colab-cli", "0.7.5"), ("jupyter-kernel-client", "0.9.0")):
            actual = importlib.metadata.version(package)
            if actual != expected:
                raise RuntimeError(f"Unsupported {package} {actual}; expected {expected}")
        from colab_cli.auth import AuthProvider
        from colab_cli.common import state

        state.auth_provider = AuthProvider.OAUTH2
        code = sys.stdin.read()  # Preserve every byte of the supplied Python text.
        session = state.get_session(args.session)

        def save_kernel(value):
            session.kernel_id = value
            state.store.add(session)

        def save_session(value):
            session.session_id = value
            state.store.add(session)

        runtime = connect_runtime(
            session.url, session.token, kernel_id=session.kernel_id, session_id=session.session_id,
            connect_timeout_s=args.connect_timeout,
            on_kernel_started=save_kernel, on_session_started=save_session,
        )
        # Prelude and user code are both inside the same cleanup scope.
        _execute_checked(runtime,
            "import os; os.makedirs('/content', exist_ok=True); os.chdir('/content')",
            min(args.connect_timeout, 10), stream=False,
        )
        session.running = "exec(stdin)"
        session.last_execution = ("stdin", None, datetime.now(timezone.utc).isoformat())
        state.store.add(session)
        print("KIT_COLAB_EXEC_READY", flush=True)
        outputs = _execute_checked(runtime, code, args.timeout, stream=True)
        state.history.log_event(args.session, "execution", {
            "code": code, "outputs": outputs, "cell_index": None, "cell_id": None,
        })
    except RemoteCodeError as error:
        failure = f"KIT_COLAB_CODE_ERROR: {_redact(str(error))}"
        result = 2
    except Exception as error:
        # Do not print a raw traceback/locals that can contain proxy credentials.
        # After READY even a timeout is ambiguous: the parent must never replay.
        failure = f"KIT_COLAB_CONNECTION_ERROR: {type(error).__name__}: {_redact(str(error))}"
        result = 3
    finally:
        try:
            if runtime is not None:
                runtime.stop()
            if session is not None and state is not None:
                session.running = None
                state.store.add(session)
        except Exception as error:
            failure = failure or f"KIT_COLAB_CONNECTION_ERROR: cleanup failed: {_redact(str(error))}"
            result = 3
        if failure:
            # The parent reaps fatal children immediately. Signal failure only
            # after socket cleanup and local running-state reconciliation.
            print(failure, flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
