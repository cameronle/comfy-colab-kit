"""Headless ComfyUI DAG executor for Google Colab batch jobs.

Runs inside the persistent Colab Jupyter kernel via `colab --auth oauth2 exec`.
- Zero HTTP/WebSocket servers, zero tunnels, zero infinite keepalive loops.
- Caches the initialized ComfyUI `PromptExecutor` in the kernel's `__main__`
  namespace (`_KIT_COMFY_ENGINE`) so consecutive jobs within the keep-warm
  window reuse VRAM-resident weights without cold-reloading 20+ GB from disk.
- Emits structured stdout lines (`KIT_JOB_PROGRESS`, `KIT_JOB_RESULT`,
  `KIT_JOB_ERROR`) consumed by `gate.batch_dispatcher`.
"""
from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_RUNTIME_CFG = "/content/kit/runtime_config.json"
KERNEL_CACHE_KEY = "_KIT_COMFY_ENGINE"


def remap_models(obj: Any, aliases: dict[str, str]) -> int:
    """Rewrite aliased model filenames in an API prompt graph."""
    if not aliases:
        return 0
    count = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v in aliases:
                obj[k] = aliases[v]
                count += 1
            else:
                count += remap_models(v, aliases)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and v in aliases:
                obj[i] = aliases[v]
                count += 1
            else:
                count += remap_models(v, aliases)
    return count


def _emit(tag: str, payload: dict[str, Any]) -> None:
    print(f"{tag} " + json.dumps(payload, ensure_ascii=False), flush=True)


def _is_oom_error(exc: BaseException | str) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg or "cublas_status_alloc_failed" in msg


def _safe_under(base_dir: Path, rel_path: str) -> Path | None:
    try:
        base_resolved = base_dir.resolve()
        candidate = (base_resolved / rel_path).resolve()
        if candidate.is_relative_to(base_resolved):
            return candidate
    except Exception:
        return None
    return None


class HeadlessPromptServer:
    """Minimal stand-in for ComfyUI's PromptServer without opening any network socket."""

    def __init__(self, prompt_id: str = ""):
        self.client_id = "kit-headless"
        self.last_node_id: str | None = None
        self.last_prompt_id: str = prompt_id
        self.loop = None
        self.sockets_metadata: dict[str, Any] = {}

    def send_sync(self, event: str, data: Any, sid: str | None = None) -> None:
        if not isinstance(data, dict):
            return
        if event == "progress_state":
            for node_id, state in (data.get("nodes") or {}).items():
                if isinstance(state, dict) and state.get("state") == "running":
                    self.send_sync("progress", {"node": node_id, "value": state.get("value"), "max": state.get("max")})
            return
        if event == "progress":
            _emit(
                "KIT_JOB_PROGRESS",
                {
                    "prompt_id": self.last_prompt_id,
                    "node": data.get("node") or self.last_node_id,
                    "step": data.get("value"),
                    "max": data.get("max"),
                },
            )
        elif event == "executing" and data.get("node"):
            self.last_node_id = str(data["node"])
            _emit(
                "KIT_JOB_PROGRESS",
                {
                    "prompt_id": self.last_prompt_id,
                    "node": self.last_node_id,
                },
            )

    def queue_updated(self) -> None:
        pass


class _RealComfyEngine:
    """Wraps ComfyUI's in-process PromptExecutor and keeps models warm across execs."""

    def __init__(self, cfg: dict[str, Any]):
        comfy_dir = Path(cfg.get("comfy_dir") or "/content/ComfyUI")
        for k, v in (cfg.get("env") or {}).items():
            os.environ[str(k)] = str(v)
        # Ensure secrets never linger in execution environment
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("CF_TUNNEL_TOKEN", None)

        if str(comfy_dir) not in sys.path:
            sys.path.insert(0, str(comfy_dir))
        os.chdir(comfy_dir)

        if cfg.get("args"):
            # Parse only profile flags, never ipykernel's argv. Keep the existing
            # Namespace object so modules that imported it see the same settings.
            cli_args = importlib.import_module("comfy.cli_args")
            args = cli_args.parser.parse_args(list(cfg["args"]), namespace=cli_args.args)
            # Match pinned cli_args' post-parse implications and normalization.
            for source, target in (
                ("high_ram", "cache_classic"), ("windows_standalone_build", "auto_launch"),
                ("windows_standalone_build", "enable_assets"), ("disable_api_nodes", "offline"),
                ("offline", "disable_partner_nodes"), ("disable_comfy_compiler", "disable_cuda_graphs"),
                ("force_fp16", "fp16_unet"), ("enable_manager_legacy_ui", "enable_manager"),
            ):
                if getattr(args, source, False):
                    setattr(args, target, True)
            if not getattr(args, "enable_assets", False):
                args.disable_assets = True
            if getattr(args, "disable_auto_launch", False):
                args.auto_launch = False
            fast = getattr(args, "fast", None)
            args.fast = set(cli_args.PerformanceFeature) if fast == [] else set(fast or ())

        emp_file = cfg.get("extra_model_paths_file")
        if emp_file and os.path.isfile(emp_file):
            try:
                extra_config = importlib.import_module("utils.extra_config")
                extra_config.load_extra_path_config(emp_file)
            except Exception:
                pass

        execution = importlib.import_module("execution")
        folder_paths = importlib.import_module("folder_paths")
        nodes = importlib.import_module("nodes")

        init_fn = getattr(nodes, "init_extra_nodes", None)
        if callable(init_fn):
            res = init_fn()
            if inspect.iscoroutine(res):
                asyncio.run(res)

        self.comfy_dir = comfy_dir
        self.execution = execution
        self.folder_paths = folder_paths
        self.server = HeadlessPromptServer()
        # bc5f880's executor unconditionally reads both RAM headroom values.
        # Keep the existing classic cache policy; zero means no RAM headroom.
        self.executor = execution.PromptExecutor(
            self.server, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0}
        )
        # main.py normally installs this hook; headless mode never imports main.
        utils = sys.modules.get("comfy.utils")
        hook_setter = getattr(utils, "set_progress_bar_global_hook", None)
        if callable(hook_setter):
            hook_setter(self._progress_hook)

    def _progress_hook(self, value, total, preview_image, prompt_id=None, node_id=None) -> None:
        # Keep only scalar progress. Do not encode preview images or open sockets.
        self.server.send_sync("progress", {"value": value, "max": total, "node": node_id or self.server.last_node_id})

    def run(self, prompt: dict[str, Any], prompt_id: str) -> dict[str, Any]:
        self.server.last_prompt_id = prompt_id
        output_dir = Path(self.folder_paths.get_output_directory())
        output_dir.mkdir(parents=True, exist_ok=True)
        before_mtimes = {
            p: p.stat().st_mtime_ns
            for p in output_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        }

        validate = self.execution.validate_prompt
        sig = inspect.signature(validate)
        if len(sig.parameters) >= 2:
            val_res = validate(prompt_id, prompt, None) if len(sig.parameters) >= 3 else validate(prompt_id, prompt)
        else:
            val_res = validate(prompt)
        if inspect.iscoroutine(val_res):
            val_res = asyncio.run(val_res)

        valid_ok, err_info, good_outputs, node_errors = val_res
        if not valid_ok:
            raise RuntimeError(f"ComfyUI prompt validation failed: {json.dumps({'error': err_info, 'node_errors': node_errors})[:600]}")

        extra_data: dict[str, Any] = {"client_id": "kit-headless"}
        if hasattr(self.executor, "execute_async") and inspect.iscoroutinefunction(self.executor.execute_async):
            asyncio.run(self.executor.execute_async(prompt, prompt_id, extra_data, good_outputs))
        else:
            self.executor.execute(prompt, prompt_id, extra_data, good_outputs)

        if not getattr(self.executor, "success", True):
            msgs = getattr(self.executor, "status_messages", [])
            raise RuntimeError(f"ComfyUI execution failed: {json.dumps(msgs)[:600]}")

        outputs_out: dict[str, dict[str, list[dict[str, str]]]] = {}
        hist = getattr(self.executor, "history_result", None) or {}
        raw_outputs = hist.get("outputs") if isinstance(hist, dict) else {}
        if isinstance(raw_outputs, dict):
            for node_id, node_out in raw_outputs.items():
                if not isinstance(node_out, dict):
                    continue
                imgs = []
                for img in node_out.get("images") or []:
                    fn = str(img.get("filename") or "")
                    sub = str(img.get("subfolder") or "").strip("/")
                    rel = f"{sub}/{fn}" if sub else fn
                    resolved = _safe_under(output_dir, rel)
                    if fn and resolved and resolved.is_file():
                        imgs.append(
                            {
                                "filename": fn,
                                "subfolder": sub,
                                "type": "output",
                                "remote_path": str(resolved),
                            }
                        )
                if imgs:
                    outputs_out[str(node_id)] = {"images": imgs}

        # Fallback: scan newly created/updated image files in output_dir
        if not outputs_out:
            fresh_imgs = []
            for p in sorted(output_dir.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                    continue
                mtime = p.stat().st_mtime_ns
                if before_mtimes.get(p) != mtime:
                    rel = p.relative_to(output_dir)
                    sub = str(rel.parent) if str(rel.parent) != "." else ""
                    fresh_imgs.append(
                        {
                            "filename": p.name,
                            "subfolder": sub,
                            "type": "output",
                            "remote_path": str(p.resolve()),
                        }
                    )
            if fresh_imgs:
                outputs_out["output"] = {"images": fresh_imgs}

        return {"outputs": outputs_out}


def _get_or_create_engine(
    cfg: dict[str, Any],
    engine_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> Any:
    main_mod = sys.modules.get("__main__")
    store = main_mod.__dict__ if main_mod is not None else globals()
    profile_id = cfg.get("profile") or "default"
    cached = store.get(KERNEL_CACHE_KEY)
    if cached is not None and cached.get("profile") != profile_id and engine_factory is None:
        raise ValueError("profile_mismatch: a different profile is already loaded in this kernel; stop and switch explicitly")
    if cached is not None and cached.get("profile") == profile_id and engine_factory is None:
        return cached["engine"]
    factory = engine_factory or _RealComfyEngine
    engine = factory(cfg)
    if engine_factory is None:
        store[KERNEL_CACHE_KEY] = {"profile": profile_id, "engine": engine}
    return engine


def execute_prompt(
    prompt: dict[str, Any],
    prompt_id: str,
    runtime_cfg_path: str = DEFAULT_RUNTIME_CFG,
    engine_factory: Callable[[dict[str, Any]], Any] | None = None,
    expected_profile: str | None = None,
) -> dict[str, Any]:
    """Execute one ComfyUI API prompt graph headlessly and return structured output info."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        # ipykernel already owns this thread's event loop. Run the whole engine
        # call on one joined worker so ComfyUI's asyncio.run calls remain valid.
        # The kernel-global engine cache still persists between consecutive jobs.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="kit-batch") as pool:
            return pool.submit(execute_prompt, prompt, prompt_id, runtime_cfg_path, engine_factory, expected_profile).result()
    if not isinstance(prompt, dict) or not prompt:
        raise ValueError("prompt must be a non-empty ComfyUI API graph dictionary")
    cfg: dict[str, Any] = {}
    if runtime_cfg_path and os.path.isfile(runtime_cfg_path):
        with open(runtime_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    if expected_profile and expected_profile != cfg.get("profile"):
        raise ValueError("profile_mismatch: task profile does not match this runtime; refusing engine initialization")
    graph = copy.deepcopy(prompt)
    remap_models(graph, cfg.get("model_aliases") or {})

    t0 = time.time()
    engine = _get_or_create_engine(cfg, engine_factory=engine_factory)
    raw = engine.run(graph, prompt_id)
    duration_s = round(time.time() - t0, 2)
    result = {
        "prompt_id": prompt_id,
        "status": "success",
        "duration_s": duration_s,
        "outputs": raw.get("outputs") or {},
    }
    _emit("KIT_JOB_RESULT", result)
    return result


def run_task_file(
    task_path: str,
    runtime_cfg_path: str = DEFAULT_RUNTIME_CFG,
    engine_factory: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Load a task JSON file, execute it, and emit KIT_JOB_RESULT or KIT_JOB_ERROR."""
    with open(task_path, "r", encoding="utf-8") as f:
        task = json.load(f)
    prompt_id = str(task.get("prompt_id") or f"job_{int(time.time())}")
    prompt = task.get("prompt")
    if isinstance(prompt, dict) and isinstance(prompt.get("prompt"), dict):
        prompt = prompt["prompt"]
    try:
        return execute_prompt(prompt, prompt_id=prompt_id, runtime_cfg_path=runtime_cfg_path,
                              engine_factory=engine_factory, expected_profile=task.get("profile_id"))
    except Exception as e:
        err_payload = {
            "prompt_id": prompt_id,
            "error": str(e)[:600],
            "oom": _is_oom_error(e),
        }
        _emit("KIT_JOB_ERROR", err_payload)
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python headless_runner.py <task.json>")
    run_task_file(sys.argv[1])
