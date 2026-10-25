"""MCP notifications must not produce unsolicited stdio replies."""
import io
import json

import pytest

from ckit.mcp_server import run_stdio_server


@pytest.mark.parametrize("method", ["notifications/initialized", "notifications/progress", "notifications/cancelled"])
def test_stdio_notifications_do_not_reply(monkeypatch, method):
    requests = [
        {"jsonrpc": "2.0", "method": method, "params": {}},
        {"jsonrpc": "2.0", "id": 0, "method": "ping"},
    ]
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr("sys.stdout", stdout)
    run_stdio_server()
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses == [{"jsonrpc": "2.0", "id": 0, "result": {}}]
