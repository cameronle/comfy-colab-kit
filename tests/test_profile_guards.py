"""Profile mismatches fail closed without stopping/reallocating a GPU."""
import json
import pytest

from gate.app import BATCH, SESSION
from gate.state import State
from tests.test_generation_profiles import client_for


@pytest.mark.parametrize("state", [State.READY, State.ALLOCATING, State.FAILED])
async def test_wake_cannot_join_another_profile(tmp_path, state):
    client = await client_for(tmp_path)
    try:
        session = client.app[SESSION]
        session.machine.profile = "moss-image"
        session.machine.state = state
        before = list(session.colab.calls)
        result = await session.wake("qwen-image")
        assert result.refused == "profile_mismatch"
        assert session.colab.calls == before
        assert session.machine.profile == "moss-image"
        assert session.machine.state == state
    finally:
        await client.close()


@pytest.mark.parametrize("path,prompt", [
    ("/_gate/txt2img", "original"),
    ("/prompt", {"1": {"class_type": "SaveImage", "inputs": {}}}),
])
async def test_http_submission_rejects_wrong_profile_without_remote_calls(tmp_path, path, prompt):
    client = await client_for(tmp_path)
    try:
        session = client.app[SESSION]
        session.machine.profile = "moss-image"
        session.machine.state = State.READY
        response = await client.post(path, json={"profile": "qwen-image", "prompt": prompt, "auto_wake": True})
        assert response.status == 409
        assert (await response.json())["error"] == "profile_mismatch"
        assert not session.colab.calls
        assert not client.app[BATCH]._jobs
    finally:
        await client.close()


def test_headless_task_checks_runtime_profile_before_engine_init(tmp_path):
    from tests.test_batch_and_security import _load_headless_runner
    hr = _load_headless_runner()
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"profile": "moss-image"}))
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"prompt_id": "sentinel", "profile_id": "qwen-image",
                                "prompt": {"1": {"class_type": "SaveImage", "inputs": {}}}}))
    calls = []

    def engine_factory(config):
        calls.append(config)
        raise AssertionError("Must not initialize any engine")

    with pytest.raises(ValueError, match="profile_mismatch"):
        hr.run_task_file(str(task), str(cfg), engine_factory=engine_factory)
    assert not calls


def test_headless_cannot_rebuild_engine_for_a_different_cached_profile(monkeypatch):
    import sys
    from tests.test_batch_and_security import _load_headless_runner
    hr = _load_headless_runner()
    monkeypatch.setitem(sys.modules["__main__"].__dict__, hr.KERNEL_CACHE_KEY,
                        {"profile": "moss-image", "engine": object()})
    calls = []

    def forbidden_init(config):
        calls.append(config)
        raise AssertionError("Must not rebuild a different model in the same kernel")

    monkeypatch.setattr(hr, "_RealComfyEngine", forbidden_init)
    with pytest.raises(ValueError, match="profile_mismatch"):
        hr._get_or_create_engine({"profile": "qwen-image"})
    assert not calls


@pytest.mark.parametrize("pid", ["qwen-image", "sd-latest"])
def test_malformed_vendored_template_fails_with_a_clear_error(tmp_path, pid):
    from dataclasses import replace
    from pathlib import Path
    from gate.generation import GenerationError, WORKFLOWS, build_generation
    from gate.profiles import load_profile
    root = Path(__file__).resolve().parents[1]
    profile = load_profile(root / "profiles/examples" / pid)
    workflow = profile.workflows[WORKFLOWS[pid]]
    template = tmp_path / workflow.file
    template.parent.mkdir(parents=True)
    template.write_text("{}")
    with pytest.raises(GenerationError, match="template"):
        build_generation(replace(profile, path=tmp_path), {"prompt": "original"})
