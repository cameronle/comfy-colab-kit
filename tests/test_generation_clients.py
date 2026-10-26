"""Client contract: let the gateway choose the profile's graph, never construct MOSS locally."""
from argparse import Namespace

import pytest
from ckit import cli, mcp_server


@pytest.mark.parametrize("profile", ["moss-image", "qwen-image", "sd-latest"])
def test_cli_submits_original_high_level_prompt_for_active_profile(monkeypatch, profile):
    calls = []

    def call(method, path, body=None):
        calls.append((method, path, body))
        if path == "/_gate/status":
            return 200, {"state": "ready", "profile": profile}
        if method == "POST":
            return 202, {"prompt_id": "client-test"}
        return 200, {"client-test": {"status": {"status_str": "success"}, "outputs": {
            "9": {"images": [{"filename": "fake.png"}]}}}}

    monkeypatch.setattr(cli, "call", call)
    monkeypatch.setattr(cli, "load_config", lambda: {})
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    args = Namespace(prompt="  unmodified\n", extra_arg=None, profile=None, workflow=None,
                     negative=None, preset=None, seed=0, width=1024, height=768,
                     timeout=180, json=True)
    cli.do_generate(args)
    submitted = [c for c in calls if c[0] == "POST"]
    assert submitted == [("POST", "/_gate/txt2img", {
        "prompt": "  unmodified\n", "profile": profile, "seed": 0,
        "width": 1024, "height": 768, "timeout_seconds": 180,
    })]


@pytest.mark.parametrize("profile", ["moss-image", "qwen-image", "sd-latest"])
def test_stdio_mcp_submits_high_level_prompt_and_exposes_profile(monkeypatch, profile):
    calls = []

    def call(method, path, body=None):
        calls.append((method, path, body))
        if path == "/_gate/status":
            return 200, {"state": "ready", "profile": profile}
        if method == "POST":
            return 202, {"prompt_id": "client-test"}
        return 200, {"client-test": {"status": {"status_str": "success"}, "outputs": {
            "9": {"images": [{"filename": "fake.png"}]}}}}

    monkeypatch.setattr(mcp_server, "call_gate", call)
    monkeypatch.setattr(mcp_server.time, "sleep", lambda _: None)
    out = mcp_server.handle_tool_call("gate_txt2img", {
        "prompt": "  unmodified\n", "profile": profile, "seed": 0,
    })
    assert not out.get("isError"), out
    submitted = [c for c in calls if c[0] == "POST"]
    assert submitted == [("POST", "/_gate/txt2img", {
        "prompt": "  unmodified\n", "profile": profile, "seed": 0,
    })]
    schema = next(t for t in mcp_server.TOOLS if t["name"] == "gate_txt2img")
    assert "profile" in schema["inputSchema"]["properties"]
    assert "Qwen" in schema["description"]


def test_stdio_mcp_can_read_profile_capabilities_without_wake(monkeypatch):
    calls = []

    def call(method, path, body=None):
        calls.append((method, path))
        return 200, {"profiles": [{"id": "qwen-image", "generation": {"presets": ["Turbo · 8 steps"]}}]}

    monkeypatch.setattr(mcp_server, "call_gate", call)
    out = mcp_server.handle_tool_call("gate_profiles", {})
    assert not out.get("isError"), out
    assert "Turbo" in out["content"][0]["text"]
    assert calls == [("GET", "/_gate/profiles")]
    assert "gate_profiles" in [t["name"] for t in mcp_server.TOOLS]
