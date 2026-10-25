"""Reproduce async Comfy execution inside a Jupyter-style running loop."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


async def test_real_engine_bridge_works_inside_running_notebook_loop(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("headless_jupyter_test", ROOT / "packages/colab/headless_runner.py")
    assert spec is not None and spec.loader is not None
    hr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hr)
    events = []

    async def init_extra_nodes():
        events.append("nodes_initialized")

    async def validate_prompt(prompt_id, prompt, partial_execution_list):
        events.append("validated")
        return True, None, ["9"], {}

    class Executor:
        success = True
        history_result = {"outputs": {}}
        def __init__(self, server, **kwargs):
            self.server = server
        async def execute_async(self, prompt, prompt_id, extra_data, outputs):
            events.append("executed")

    monkeypatch.setitem(sys.modules, "execution", SimpleNamespace(validate_prompt=validate_prompt, PromptExecutor=Executor))
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_output_directory=lambda: str(tmp_path / "out")))
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(init_extra_nodes=init_extra_nodes))
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "runtime.json"
    cfg.write_text(json.dumps({"comfy_dir": str(tmp_path), "profile": "async-regression"}))
    main = sys.modules["__main__"].__dict__
    old = main.pop(hr.KERNEL_CACHE_KEY, None)
    try:
        # pytest-asyncio has a real running loop, as ipykernel does.
        assert asyncio.get_running_loop().is_running()
        result = hr.execute_prompt({"9": {"class_type": "SaveImage", "inputs": {}}}, "jupyter-test", str(cfg))
        assert result["status"] == "success"
        assert events == ["nodes_initialized", "validated", "executed"]
    finally:
        main.pop(hr.KERNEL_CACHE_KEY, None)
        if old is not None:
            main[hr.KERNEL_CACHE_KEY] = old
