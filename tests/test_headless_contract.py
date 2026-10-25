"""Pinned-ComfyUI adapter contracts, without importing torch or allocating GPU."""
import argparse
import importlib.util
import json
import sys
import pytest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


def load_runner():
    spec = importlib.util.spec_from_file_location("headless_contract_test", ROOT / "packages/colab/headless_runner.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("fast_flag", [[], ["--fast"]])
def test_engine_applies_profile_flags_before_constructing_executor(tmp_path, monkeypatch, fast_flag):
    hr = load_runner()
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve-vram", type=float)
    parser.add_argument("--preview-method", default="none")
    parser.add_argument("--fast", nargs="*")
    args = parser.parse_args([])
    args.fast = set()
    cli_args = SimpleNamespace(parser=parser, args=args, PerformanceFeature=("fp16_accumulation",))
    hooks = []

    class Executor:
        def __init__(self, server, **kwargs):
            assert args.reserve_vram == 1.0
            assert args.preview_method == "auto"
            assert isinstance(args.fast, set)
            assert args.fast == ({"fp16_accumulation"} if fast_flag else set())

    monkeypatch.setitem(sys.modules, "comfy.cli_args", cli_args)
    monkeypatch.setitem(sys.modules, "execution", SimpleNamespace(PromptExecutor=Executor))
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "comfy.utils", SimpleNamespace(set_progress_bar_global_hook=hooks.append))
    monkeypatch.chdir(tmp_path)
    engine = hr._RealComfyEngine({"comfy_dir": str(tmp_path), "args": ["--reserve-vram", "1", "--preview-method", "auto"] + fast_flag})
    assert len(hooks) == 1
    engine.server.last_node_id = "3"
    hooks[0](4, 10, None)


def test_headless_server_accepts_pinned_progress_state_events(capsys):
    hr = load_runner()
    server = hr.HeadlessPromptServer("job-test")
    server.send_sync("progress_state", {"nodes": {
        "3": {"node_id": "3", "value": 4, "max": 10, "state": "running"},
    }})
    lines = capsys.readouterr().out.splitlines()
    progress = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("KIT_JOB_PROGRESS ")]
    assert progress == [{"prompt_id": "job-test", "node": "3", "step": 4, "max": 10}]
    # Preview feature checks in the pinned ProgressHandler read this field.
    assert server.sockets_metadata == {}
