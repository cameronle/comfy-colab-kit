"""Input discovery regressions; use a fake runtime, never Google compute."""
import asyncio
import json
import os
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gate.app import BATCH, build_app
from gate.batch_dispatcher import BatchDispatcher, BatchError
from gate.config import Settings
from gate.profiles import load_profile
from gate.session import Session
from tests.fakes import FAKE_PNG, FakeColab

ROOT = Path(__file__).resolve().parent.parent


async def test_long_chinese_prompt_is_not_a_filename_and_reaches_runner_unchanged(tmp_path):
    text = '中文提示词' * 25
    assert len(text) <= 160 and len(text.encode('utf-8')) > 255
    settings = Settings.from_env({
        'AUTH_DISABLED': 'true', 'STATE_DIR': str(tmp_path), 'DEFAULT_PROFILE': 'tiny',
    })
    remote = FakeColab()
    app = build_app(settings, colab=remote, profiles={
        'tiny': load_profile(ROOT / 'tests/fixtures/profiles/tiny'),
    }, run_background=False)
    client = TestClient(TestServer(app))
    await client.start_server()
    graph = {
        '1': {'class_type': 'CLIPTextEncode', 'inputs': {'text': text}},
        '2': {'class_type': 'SaveImage', 'inputs': {}},
    }
    try:
        response = await client.post('/prompt', json={'prompt': graph, 'auto_wake': True})
        assert response.status == 200
        job = app[BATCH].get_job((await response.json())['prompt_id'])
        await asyncio.wait_for(job.done_event.wait(), timeout=5)
        assert job.status == 'success', job.error
        task = json.loads(remote.remote_files[f'/content/kit/tasks/{job.prompt_id}.json'])
        assert task['prompt'] == graph
        assert not any(call.startswith('upload:/content/ComfyUI/input/') for call in remote.calls)
    finally:
        await client.close()


@pytest.fixture
def input_dispatcher(tmp_path):
    settings = Settings.from_env({'AUTH_DISABLED': 'true', 'STATE_DIR': str(tmp_path)})
    remote = FakeColab()
    session = Session(settings, {}, remote)
    session.machine.session_started = 123.0
    return BatchDispatcher(settings, session), remote


async def test_input_discovery_ignores_existing_non_image_strings(input_dispatcher):
    dispatcher, remote = input_dispatcher
    (dispatcher.inputs_dir / 'reference.png').write_bytes(FAKE_PNG)
    (dispatcher.inputs_dir / 'unused.png').write_bytes(FAKE_PNG)
    (dispatcher.inputs_dir / 'notes.txt').write_text('synthetic non-image data')
    await dispatcher._upload_referenced_inputs('owned-test-session', {
        '1': {'inputs': {'text': 'notes.txt', 'image': 'reference.png'}},
    })
    assert remote.remote_files == {'/content/ComfyUI/input/reference.png': FAKE_PNG}


async def test_input_reference_cannot_follow_symlink_outside_storage(input_dispatcher, tmp_path):
    dispatcher, remote = input_dispatcher
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'synthetic-outside-storage')
    (dispatcher.inputs_dir / 'linked.png').symlink_to(outside)
    with pytest.raises(BatchError, match='escapes storage'):
        await dispatcher._upload_referenced_inputs('owned-test-session', {'image': 'linked.png'})
    assert not remote.remote_files


async def test_unmatched_prompt_strings_never_trigger_file_probes(input_dispatcher, monkeypatch):
    dispatcher, remote = input_dispatcher
    (dispatcher.inputs_dir / 'reference.png').write_bytes(FAKE_PNG)
    original = Path.is_file
    checked = []

    def guard(path):
        if path.parent == dispatcher.inputs_dir:
            checked.append(path.name)
            assert path.name == 'reference.png', 'Prompt text reached the filesystem'
        return original(path)

    monkeypatch.setattr(Path, 'is_file', guard)
    await dispatcher._upload_referenced_inputs('owned-test-session', {
        'text': '中文提示词' * 25,
        'negative': 'a' * 160,
        'other': ['missing.png', 'invalid\x00.png', '../outside.png', 'text' * 1000],
        'nested': {'image': 'reference.png'},
    })
    assert checked and set(checked) == {'reference.png'}


@pytest.mark.parametrize('filename', ['reference.png', 'photo.JPG', '图像.webp'])
async def test_uploaded_images_keep_mtime_cache_and_reupload_for_new_vm(input_dispatcher, filename):
    dispatcher, remote = input_dispatcher
    saved = dispatcher.save_input_image(filename, FAKE_PNG)
    name = saved['name']
    graph = {'1': {'inputs': {'nested': [name, name]}}}
    target = f'/content/ComfyUI/input/{name}'
    for _ in range(2):
        await dispatcher._upload_referenced_inputs('owned-test-session', graph)
    assert remote.calls.count('upload:' + target) == 1
    image = dispatcher.inputs_dir / name
    before = image.stat().st_mtime_ns
    image.write_bytes(FAKE_PNG + b'updated')
    os.utime(image, ns=(before + 1_000_000, before + 1_000_000))
    await dispatcher._upload_referenced_inputs('owned-test-session', graph)
    assert remote.calls.count('upload:' + target) == 2
    dispatcher.session.machine.session_started = 456.0
    await dispatcher._upload_referenced_inputs('owned-test-session', graph)
    assert remote.calls.count('upload:' + target) == 3
    assert remote.remote_files[target] == FAKE_PNG + b'updated'
