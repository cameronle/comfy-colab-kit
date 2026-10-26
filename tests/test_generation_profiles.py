"""Multi-profile quick-generation contracts: local fakes, never real Colab."""
import asyncio
import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gate.app import BATCH, SESSION, build_app
from gate.config import Settings
from gate.profiles import load_profile
from tests.fakes import FakeColab

ROOT = Path(__file__).resolve().parents[1]


async def client_for(tmp_path, profile="moss-image", colab=None):
    settings = Settings.from_env({
        "AUTH_DISABLED": "true", "STATE_DIR": str(tmp_path),
        "DEFAULT_PROFILE": profile,
    })
    profiles = {name: load_profile(ROOT / "profiles/examples" / name)
                for name in ("moss-image", "qwen-image", "sd-latest")}
    app = build_app(settings, colab=colab or FakeColab(), profiles=profiles, run_background=False)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def completed(client, response):
    assert response.status == 202, await response.text()
    payload = await response.json()
    job = client.app[BATCH].get_job(payload["prompt_id"])
    await asyncio.wait_for(job.done_event.wait(), 5)
    assert job.status == "success", job.error
    return job


async def test_moss_quick_generation_preserves_prompt_and_records_model(tmp_path):
    client = await client_for(tmp_path)
    text = "  水豚穿着西装，会议室里的长提示词。" * 90 + "\n  "
    try:
        job = await completed(client, await client.post("/_gate/txt2img", json={
            "prompt": text, "profile": "moss-image", "auto_wake": True,
            "seed": 0, "width": 1024, "height": 768,
            "preset": "Fast alt", "negative_prompt": "",
        }))
        assert job.prompt["6"]["inputs"]["value"] == text
        assert job.prompt["8"]["inputs"]["text"] == ""
        summary = job.to_summary()
        assert summary["profile_id"] == "moss-image"
        assert summary["workflow_id"] == "matlowai-presets"
        assert summary["parameters"]["seed"] == 0
        assert summary["parameters"]["steps"] == 10
        stored = json.loads((tmp_path / "jobs.json").read_text())[-1]
        assert stored["profile_id"] == "moss-image"
        assert stored["prompt"] == job.prompt
    finally:
        await client.close()


async def test_qwen_turbo_uses_its_official_conditioning_cache_and_schedule(tmp_path):
    client = await client_for(tmp_path, "qwen-image")
    text = "  中文海报，标题为‘原样保留’。\n" * 60
    try:
        job = await completed(client, await client.post("/_gate/txt2img", json={
            "prompt": text, "profile": "qwen-image", "auto_wake": True, "seed": 42,
        }))
        graph = job.prompt
        assert graph["451"]["inputs"]["unet_name"] == "qwen_image_2.1_turbo_int8_convrot.safetensors"
        assert graph["452"]["class_type"] == "TextEncodeQwenImage21"
        assert graph["490"]["class_type"] == "QwenImage21Cache"
        assert graph["474"]["inputs"]["on_false"] == text
        assert graph["474"]["inputs"]["switch"] is False  # No automatic prompt rewriting.
        assert graph["488"]["inputs"]["noise_seed"] == 42
        assert graph["489"]["inputs"]["cfg"] == 1
        sigmas = [float(x.strip()) for x in graph["484"]["inputs"]["sigmas"].split(",")]
        assert sigmas == [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0]
        assert graph["461"]["class_type"] == "SaveImageAdvanced"
        assert graph["461"]["inputs"]["format.bit_depth"] == "8-bit"
        assert job.to_summary()["profile_id"] == "qwen-image"
        assert job.to_summary()["workflow_id"] == "qwen21-turbo-t2i"
        assert job.to_summary()["parameters"]["steps"] == 8
    finally:
        await client.close()


async def test_sd35_uses_its_checkpoint_conditioning_and_20_step_sampler(tmp_path):
    client = await client_for(tmp_path, "sd-latest")
    try:
        job = await completed(client, await client.post("/_gate/txt2img", json={
            "prompt": "  original prompt\n", "negative_prompt": "blur", "profile": "sd-latest",
            "auto_wake": True, "seed": 17, "width": 1024, "height": 768,
        }))
        graph = job.prompt
        assert graph["4"]["inputs"]["ckpt_name"] == "sd3.5_large_fp8_scaled.safetensors"
        assert graph["16"]["inputs"]["text"] == "  original prompt\n"
        assert graph["40"]["inputs"]["text"] == "blur"
        assert graph["53"]["class_type"] == "EmptySD3LatentImage"
        assert graph["53"]["inputs"] == {"width": 1024, "height": 768, "batch_size": 1}
        assert graph["3"]["inputs"] == {
            "model": ["4", 0], "positive": ["16", 0], "negative": ["40", 0],
            "latent_image": ["53", 0], "seed": 17, "steps": 20, "cfg": 4.01,
            "sampler_name": "euler", "scheduler": "sgm_uniform", "denoise": 1,
        }
        assert graph["9"]["class_type"] == "SaveImage"
        assert job.to_summary()["profile_id"] == "sd-latest"
        assert job.to_summary()["workflow_id"] == "sd35-large"
    finally:
        await client.close()


async def test_catalog_exposes_capabilities_per_model_without_allocating(tmp_path):
    client = await client_for(tmp_path)
    try:
        catalog = await (await client.get("/_gate/profiles")).json()
        profiles = {p["id"]: p for p in catalog["profiles"]}
        assert len(profiles) == 3
        assert profiles["moss-image"]["generation"]["reference_image"] is True
        qwen = profiles["qwen-image"]["generation"]
        assert qwen["presets"] == ["Turbo · 8 steps"]
        assert qwen["negative_prompt"] is False
        assert qwen["reference_image"] is False
        assert "non-commercial" in qwen["license"]
        assert profiles["sd-latest"]["generation"]["presets"] == ["Quality · 20 steps"]
        mcp = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                                "params": {"name": "gate_profiles", "arguments": {}}})
        result = (await mcp.json())["result"]
        assert not result.get("isError"), result
        assert json.loads(result["content"][0]["text"])["profiles"] == catalog["profiles"]
        listing = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = (await listing.json())["result"]["tools"]
        assert "gate_profiles" in {t["name"] for t in tools}
        from ckit.mcp_server import TOOLS
        assert next(t for t in tools if t["name"] == "gate_txt2img") == next(t for t in TOOLS if t["name"] == "gate_txt2img")
        assert not client.app[SESSION].colab.calls
    finally:
        await client.close()


@pytest.mark.parametrize("profile,node,field", [
    ("moss-image", "6", "value"), ("qwen-image", "474", "on_false"),
    ("sd-latest", "16", "text"),
])
async def test_http_mcp_uses_same_profile_adapter(tmp_path, profile, node, field):
    client = await client_for(tmp_path, profile)
    try:
        await client.post("/_gate/wake", json={"profile": profile})
        for _ in range(100):
            if client.app[SESSION].machine.state.value == "ready":
                break
            await asyncio.sleep(0.01)
        response = await client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "gate_txt2img", "arguments": {
                "prompt": "  exact original\n", "profile": profile, "seed": 0,
            }},
        })
        result = (await response.json())["result"]
        assert not result.get("isError"), result
        job = next(iter(client.app[BATCH]._jobs.values()))
        assert job.prompt[node]["inputs"][field] == "  exact original\n"
        assert job.profile_id == profile
    finally:
        await client.close()


async def test_switch_refuses_while_queue_busy_and_does_not_stop_vm(tmp_path):
    from gate.state import State
    client = await client_for(tmp_path)
    try:
        session = client.app[SESSION]
        session.machine.profile = "moss-image"
        session.machine.state = State.READY
        session.queue_busy = True
        result = await session.switch("qwen-image")
        assert result.refused == "queue_busy"
        assert session.machine.profile == "moss-image"
        assert session.machine.state == State.READY
        assert "stop" not in session.colab.calls
    finally:
        await client.close()


async def test_worker_rechecks_job_profile_before_remote_upload(tmp_path):
    from gate.batch_dispatcher import BatchJob
    from gate.state import State
    client = await client_for(tmp_path)
    try:
        session = client.app[SESSION]
        session.machine.profile = "moss-image"
        session.machine.state = State.READY
        job = BatchJob("test", 1, {"1": {"class_type": "SaveImage", "inputs": {}}}, "test", 0,
                       profile_id="qwen-image", workflow_id="qwen21-turbo-t2i")
        await client.app[BATCH]._execute_job(job)
        assert job.status == "error"
        assert "profile" in job.error.lower()
        assert not session.colab.calls
    finally:
        await client.close()


@pytest.mark.parametrize("profile,extra,error", [
    ("moss-image", {"denoise": 0.5}, "unsupported_parameter"),
    ("qwen-image", {"negative_prompt": "blur"}, "unsupported_parameter"),
    ("qwen-image", {"reference_image": "input.png"}, "unsupported_parameter"),
    ("sd-latest", {"denoise": 0.5}, "unsupported_parameter"),
    ("sd-latest", {"preset": "Fast alt"}, "unsupported_preset"),
    ("qwen-image", {"workflow": "qwen21-edit"}, "unsupported_workflow"),
    ("moss-image", {"steps": 50}, "unsupported_parameter"),
    ("moss-image", {"seed": True}, "invalid_parameter"),
    ("qwen-image", {"width": 1000}, "invalid_dimensions"),
    ("sd-latest", {"height": 777}, "invalid_dimensions"),
])
async def test_invalid_generation_is_rejected_before_auto_wake(tmp_path, profile, extra, error):
    client = await client_for(tmp_path, profile)
    try:
        response = await client.post("/_gate/txt2img", json={
            "prompt": "unaltered", "profile": profile, "auto_wake": True, **extra,
        })
        assert response.status == 400, await response.text()
        assert (await response.json())["error"] == error
        assert not client.app[SESSION].colab.calls
        assert not client.app[BATCH]._jobs
    finally:
        await client.close()
