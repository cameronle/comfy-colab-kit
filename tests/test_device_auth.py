"""Tests for SQLite SHA-256 hashed tokens and RFC 8628 Device Authorization flow."""
import sqlite3
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gate.app import build_app
from gate.config import Settings
from gate.db import GateDB
from gate.profiles import load_profile
from tests.fakes import FakeColab

ROOT = Path(__file__).resolve().parent.parent


def test_gate_db_device_flow_zero_plaintext(tmp_path):
    db_file = tmp_path / "test_gate.db"
    db = GateDB(db_file)

    code_info = db.create_device_code(client_name="Test-Cursor", expires_in_sec=60)
    dev_code = code_info["device_code"]
    user_code = code_info["user_code"]
    assert dev_code.startswith("dev_") and "-" in user_code

    assert db.poll_device_token(dev_code)["status"] == "pending"

    app_res = db.approve_device_code(user_code, email="alice@example.com", token_name="Cursor-Mac")
    assert "token" not in app_res  # Not minted until consumed in memory

    poll2 = db.poll_device_token(dev_code)
    assert poll2["status"] == "approved"
    raw_token = poll2["access_token"]
    assert raw_token.startswith("sk_agent_")
    assert poll2["email"] == "alice@example.com"

    # Re-polling an already consumed device code must not return another token
    assert db.poll_device_token(dev_code)["status"] == "consumed"

    # Verify raw token is NEVER stored anywhere in SQLite on disk
    db_bytes = db_file.read_bytes() + (
        (tmp_path / "test_gate.db-wal").read_bytes() if (tmp_path / "test_gate.db-wal").exists() else b""
    )
    assert raw_token.encode() not in db_bytes
    with sqlite3.connect(db_file) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(agent_tokens)").fetchall()}
        assert "token_plain" not in cols

    verified = db.verify_token(raw_token)
    assert verified is not None and verified["email"] == "alice@example.com"

    tokens = db.list_tokens(email="alice@example.com")
    assert len(tokens) == 1
    assert "token" not in tokens[0]
    assert tokens[0]["token_prefix"] == raw_token[:15]

    assert db.revoke_token(tokens[0]["id"], email="alice@example.com") is True
    assert db.verify_token(raw_token) is None


@pytest.mark.asyncio
async def test_device_auth_endpoints_and_security_guards(tmp_path):
    settings = Settings.from_env({
        "STATE_DIR": str(tmp_path),
        "AUTH_DISABLED": "true",
        "OWNER_EMAILS": "admin@example.com",
        "DEFAULT_PROFILE": "tiny",
    })
    prof = {"tiny": load_profile(ROOT / "tests/fixtures/profiles/tiny")}
    app = build_app(settings, colab=FakeColab(), profiles=prof, run_background=False)
    client = TestClient(TestServer(app))
    await client.start_server()

    try:
        # 1. Reflected XSS payload in /auth/device?code= must be HTML-escaped
        xss = '"><svg onload=alert(1)>'
        html_res = await client.get("/auth/device", params={"code": xss})
        html_text = await html_res.text()
        assert html_res.status == 200
        assert "<svg onload=" not in html_text.lower()
        assert "&quot;&gt;&lt;SVG" in html_text

        # 2. Device code pairing
        res = await client.post("/api/auth/device/code", json={"client_name": "Cursor"})
        data = await res.json()
        user_code, dev_code = data["user_code"], data["device_code"]

        res = await client.post("/api/auth/device/token", json={"device_code": dev_code})
        assert res.status == 400 and (await res.json())["error"] == "authorization_pending"

        res = await client.post("/api/auth/device/approve", json={"user_code": user_code})
        assert res.status == 200 and (await res.json())["ok"] is True

        res = await client.post("/api/auth/device/token", json={"device_code": dev_code})
        assert res.status == 200
        token = (await res.json())["access_token"]

        # 3. Bearer token works for status, but CANNOT mint replacement tokens or approve device codes!
        agent_hdr = {"Authorization": f"Bearer {token}"}
        assert (await client.get("/_gate/status", headers=agent_hdr)).status == 200

        mint_attempt = await client.post("/api/agent-tokens", json={"name": "backdoor"}, headers=agent_hdr)
        assert mint_attempt.status == 403
        assert (await mint_attempt.json())["error"] == "human_auth_required"

        list_attempt = await client.get("/api/agent-tokens", headers=agent_hdr)
        assert list_attempt.status == 403

        # 4. Human session can list metadata (no plaintext) and revoke
        res = await client.get("/api/agent-tokens")
        toks = (await res.json())["tokens"]
        assert len(toks) == 1 and "token" not in toks[0]
        tid = toks[0]["id"]

        res = await client.post(f"/api/agent-tokens/{tid}/revoke")
        assert res.status == 200
        assert (await client.get("/_gate/status", headers=agent_hdr)).status == 401

        # 5. Direct one-time token issuance from human browser session
        res = await client.post("/api/agent-tokens", json={"name": "Hermes-Agent"})
        assert res.status == 200
        dir_data = await res.json()
        assert dir_data["token"].startswith("sk_agent_") and dir_data["one_time"] is True
        assert (await client.get("/_gate/status", headers={"Authorization": f"Bearer {dir_data['token']}"})).status == 200
    finally:
        await client.close()
