"""Stateless Streamable HTTP contract, without allocating cloud resources."""
from aiohttp.test_utils import TestClient, TestServer
import pytest

from gate.app import build_app
from gate.config import Settings
from tests.fakes import FakeColab


@pytest.mark.parametrize("version", ["2025-03-26", "2025-06-18"])
async def test_mcp_negotiates_supported_http_protocol(tmp_path, version):
    app = build_app(Settings.from_env({"AUTH_DISABLED": "true", "STATE_DIR": str(tmp_path)}), FakeColab(), {}, run_background=False)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": version, "capabilities": {}, "clientInfo": {"name": "transport-test", "version": "1"}}})
        assert response.status == 200
        assert (await response.json())["result"]["protocolVersion"] == version


async def test_mcp_initialized_notification_returns_202_without_body(tmp_path):
    app = build_app(Settings.from_env({"AUTH_DISABLED": "true", "STATE_DIR": str(tmp_path)}), FakeColab(), {}, run_background=False)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert response.status == 202
        assert await response.read() == b""


async def test_mcp_cross_origin_browser_request_is_rejected(tmp_path):
    app = build_app(Settings.from_env({"AUTH_DISABLED": "true", "STATE_DIR": str(tmp_path), "PUBLIC_HOST": "comfy.example.com"}), FakeColab(), {}, run_background=False)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/mcp", headers={"Origin": "https://evil.example"}, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert response.status == 403
