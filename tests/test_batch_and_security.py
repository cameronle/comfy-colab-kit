"""End-to-end tests for compliant headless batch execution and security controls."""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer
from yarl import URL

from gate.app import BATCH, SESSION, build_app
from gate.colab import ColabCLI
from gate.config import Settings
from gate.db import DB_KEY
from gate.profiles import load_profile
from gate.state import State
from tests.fakes import FAKE_PNG, FakeColab

ROOT = Path(__file__).resolve().parent.parent


def _load_headless_runner():
    p = ROOT / "packages/colab/headless_runner.py"
    spec = importlib.util.spec_from_file_location("kit_headless_runner_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def test_mcp_wake_enforces_owner_only_authorization(tmp_path):
    """Regression test: non-owner bearer token must not bypass OWNER_ONLY_CONTROL via /mcp."""
    settings = Settings.from_env({
        "CF_ACCESS_TEAM_DOMAIN": "test.cloudflareaccess.com",
        "CF_ACCESS_AUD": "aud-1",
        "OWNER_EMAILS": "owner@example.com",
        "ALLOWED_EMAILS": "guest@example.com,owner@example.com",
        "OWNER_ONLY_CONTROL": "true",
        "STATE_DIR": str(tmp_path),
        "DEFAULT_PROFILE": "tiny",
    })
    colab = FakeColab()
    prof = {"tiny": load_profile(ROOT / "tests/fixtures/profiles/tiny")}
    app = build_app(settings, colab=colab, profiles=prof, run_background=False)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        guest = app[DB_KEY].create_direct_token("guest@example.com", "guest-cli")
        headers = {"Authorization": f"Bearer {guest['token']}"}

        r_rest = await client.post("/_gate/wake", json={"profile": "tiny"}, headers=headers)
        assert r_rest.status == 403

        r_mcp = await client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "gate_wake", "arguments": {"profile": "tiny"}}},
            headers=headers,
        )
        assert r_mcp.status == 403
        body = await r_mcp.json()
        assert body["result"]["isError"] is True
        assert not any(c.startswith("new:") for c in colab.calls)
        assert app[SESSION].machine.state == State.SLEEPING
    finally:
        await client.close()


async def test_colab_cli_always_passes_explicit_oauth2_flag(monkeypatch):
    seen_argv = []

    class DummyProc:
        returncode = 0

        async def communicate(self, input=None):
            return b"Current balance: 42.5 compute units\n", b""

    async def fake_exec(*argv, **kwargs):
        seen_argv.append(argv)
        return DummyProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    cli = ColabCLI("/home/hermes/.local/bin/colab")
    bal = await cli.ccu_info()
    assert bal == 42.5
    assert seen_argv[0][:4] == ("/home/hermes/.local/bin/colab", "--auth", "oauth2", "usage")


async def test_batch_pipeline_upload_txt2img_mcp_and_rolling_cleanup(tmp_path):
    settings = Settings.from_env({
        "AUTH_DISABLED": "true",
        "STATE_DIR": str(tmp_path),
        "DEFAULT_PROFILE": "tiny",
        "MAX_OUTPUT_FILES": "2",
    })
    colab = FakeColab()
    prof = {"tiny": load_profile(ROOT / "tests/fixtures/profiles/tiny")}
    app = build_app(settings, colab=colab, profiles=prof, run_background=False)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        # 1. Upload a reference image for img2img
        form = aiohttp.FormData()
        form.add_field("image", FAKE_PNG, filename="ref photo.png", content_type="image/png")
        up_res = await client.post("/upload/image", data=form)
        assert up_res.status == 200
        up_data = await up_res.json()
        ref_name = up_data["name"]
        assert ref_name == "ref_photo.png"

        # 2. Submit txt2img + reference_image with auto_wake=True while sleeping
        gen_res = await client.post(
            "/_gate/txt2img",
            json={
                "prompt": "clean flat minimalist icon",
                "preset": "Fast alt",
                "reference_image": ref_name,
                "denoise": 0.55,
                "auto_wake": True,
            },
        )
        assert gen_res.status == 202
        pid1 = (await gen_res.json())["prompt_id"]

        # Wait for job completion
        job1 = app[BATCH].get_job(pid1)
        await asyncio.wait_for(job1.done_event.wait(), timeout=5)
        assert job1.status == "success"
        assert f"upload:/content/ComfyUI/input/{ref_name}" in colab.calls

        # 3. Verify ComfyUI-compatible /history and /view
        hist = await (await client.get(f"/history/{pid1}")).json()
        assert pid1 in hist
        out_fn1 = hist[pid1]["outputs"]["26"]["images"][0]["filename"]
        view_res = await client.get(f"/view?filename={out_fn1}&type=output")
        assert view_res.status == 200
        assert await view_res.read() == FAKE_PNG

        # 4. Path traversal attempts on /view must be blocked
        for bad in ("../session.json", "..%2Fsession.json", "/etc/passwd", ".hidden.png"):
            bad_res = await client.get(URL(f"/view?filename={bad}&type=output", encoded=True))
            assert bad_res.status == 400

        # 5. Execute two more jobs via HTTP /mcp (`gate_txt2img` and `gate_generate`) and verify MAX_OUTPUT_FILES=2 pruning
        mcp_r1 = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": "gate_txt2img", "arguments": {"prompt": "second job via mcp"}},
            },
        )
        assert mcp_r1.status == 200
        mcp_body1 = await mcp_r1.json()
        assert not mcp_body1["result"].get("isError")
        assert "Generated Images" in mcp_body1["result"]["content"][0]["text"]

        mcp_r2 = await client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "gate_generate",
                    "arguments": {"prompt": {"1": {"class_type": "SaveImage", "inputs": {}}}},
                },
            },
        )
        assert mcp_r2.status == 200
        assert not (await mcp_r2.json())["result"].get("isError")

        output_files = list((tmp_path / "outputs").glob("*.png"))
        assert len(output_files) == 2

        # 6. CUDA OOM in batch execution triggers colab.restart_kernel automatically
        oom_res = await client.post(
            "/prompt",
            json={"prompt": {"1": {"class_type": "SaveImage", "inputs": {"tag": "FORCE_CUDA_OOM"}}}},
        )
        oom_pid = (await oom_res.json())["prompt_id"]
        oom_job = app[BATCH].get_job(oom_pid)
        await asyncio.wait_for(oom_job.done_event.wait(), timeout=5)
        assert oom_job.status == "error"
        assert "restart_kernel" in colab.calls
    finally:
        await client.close()


def test_headless_runner_warm_singleton_and_alias_remap(tmp_path):
    hr = _load_headless_runner()
    main_dict = sys.modules["__main__"].__dict__
    main_dict.pop(hr.KERNEL_CACHE_KEY, None)

    cfg_path = tmp_path / "runtime_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "profile": "moss-image",
                "model_aliases": {"aliased.safetensors": "real_shared.safetensors"},
            }
        )
    )

    seen_graphs = []
    init_count = [0]

    class FakeEngine:
        def __init__(self, cfg):
            init_count[0] += 1

        def run(self, graph, prompt_id):
            seen_graphs.append(graph)
            out_img = tmp_path / f"{prompt_id}.png"
            out_img.write_bytes(FAKE_PNG)
            return {
                "outputs": {
                    "9": {
                        "images": [
                            {
                                "filename": out_img.name,
                                "subfolder": "",
                                "type": "output",
                                "remote_path": str(out_img),
                            }
                        ]
                    }
                }
            }

    # Test warm singleton caching via monkeypatched _RealComfyEngine
    orig_engine = hr._RealComfyEngine
    hr._RealComfyEngine = FakeEngine
    try:
        task1 = tmp_path / "task1.json"
        task1.write_text(
            json.dumps(
                {
                    "prompt_id": "p1",
                    "prompt": {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "aliased.safetensors"}}},
                }
            )
        )
        res1 = hr.run_task_file(str(task1), runtime_cfg_path=str(cfg_path))
        res2 = hr.run_task_file(str(task1), runtime_cfg_path=str(cfg_path))
        assert init_count[0] == 1  # Engine initialized once and kept warm in kernel!
        assert seen_graphs[0]["1"]["inputs"]["unet_name"] == "real_shared.safetensors"
        assert res1["status"] == "success" and res2["status"] == "success"
    finally:
        hr._RealComfyEngine = orig_engine
        main_dict.pop(hr.KERNEL_CACHE_KEY, None)
