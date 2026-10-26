"""Validated, profile-aware quick generation shared by every gateway client.

No network, model downloads or GPU imports are allowed in this module.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import secrets
from typing import Any

from ckit.moss_template import PRESETS, make_moss_txt2img_graph

from .profiles import Profile

DEFAULT_NEGATIVE = "worst quality, bad anatomy, watermark, jpeg artifact"
WORKFLOWS = {"moss-image": "matlowai-presets", "qwen-image": "qwen21-turbo-t2i", "sd-latest": "sd35-large"}
MOSS_ALIASES = {
    "quality": 0, "quality+": 1, "balanced": 2,
    "fast": 3, "fast alt": 4, "fast-alt": 4, "off": 5, "plain": 5,
}


class GenerationError(ValueError):
    def __init__(self, status: int, code: str, msg: str):
        super().__init__(msg)
        self.status, self.code, self.msg = status, code, msg


@dataclass(frozen=True)
class Generation:
    graph: dict
    profile_id: str
    workflow_id: str
    preset: str
    parameters: dict
    timeout_s: int


def _integer(body: dict, key: str, default: int, low: int, high: int) -> int:
    value = body.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise GenerationError(400, "invalid_parameter", f"{key} must be an integer between {low} and {high}")
    return value


def capabilities(profile: Profile) -> dict:
    workflow_id = WORKFLOWS.get(profile.id)
    if workflow_id not in profile.workflows:
        return {"available": False, "reason": "No quick-generation adapter for this profile"}
    if profile.id == "sd-latest":
        return {
            "available": True, "workflow_id": workflow_id,
            "presets": ["Quality · 20 steps"], "default_preset": "Quality · 20 steps",
            "negative_prompt": True, "reference_image": False, "default_negative": "",
            "default_width": 1024, "default_height": 1024, "dimension_multiple": 16,
            "sizes": [[1024, 1024], [1024, 768], [768, 1024], [1344, 768]],
            "license": "Stability AI Community License; commercial use is subject to its revenue and other conditions",
        }
    if profile.id == "qwen-image":
        return {
            "available": True, "workflow_id": workflow_id,
            "presets": ["Turbo · 8 steps"], "default_preset": "Turbo · 8 steps",
            "negative_prompt": False, "reference_image": False, "default_negative": "",
            "default_width": 1024, "default_height": 1024, "dimension_multiple": 32,
            "sizes": [[1024, 1024], [1024, 768], [768, 1024], [1344, 768]],
            "license": "Qwen Research License: non-commercial research/evaluation only; commercial use needs separate permission",
        }
    return {
        "available": True, "workflow_id": workflow_id,
        "presets": [PRESETS[4], PRESETS[3], PRESETS[2], PRESETS[0], PRESETS[1], PRESETS[5]],
        "default_preset": PRESETS[4], "negative_prompt": True,
        "reference_image": True, "default_negative": DEFAULT_NEGATIVE,
        "default_width": 1024, "default_height": 1024, "dimension_multiple": 16,
        "sizes": [[1024, 1024], [1024, 768], [768, 1024], [1344, 768]],
        "license": "Apache-2.0 (default model and adapters)",
    }


def _reference(graph: dict, filename: str, denoise: float) -> None:
    graph["90"] = {"class_type": "LoadImage", "inputs": {"image": filename}}
    graph["91"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["90", 0], "vae": ["5", 0]}}
    for nid, node in graph.items():
        if nid in ("90", "91"):
            continue
        for key, value in list(node["inputs"].items()):
            if value == ["23", 0]:
                node["inputs"][key] = ["91", 0]
            if key == "denoise":
                node["inputs"][key] = denoise
    graph.pop("23", None)


# Explicit widget contracts for the pinned templates, not a heuristic converter.
# Frontend-only seed control widgets are deliberately excluded from API inputs.
_WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "VAELoader": ["vae_name"],
    "TextEncodeQwenImage21": ["prompt", "negative_prompt", "resolution"],
    "EmptyLatentImage": ["width", "height", "batch_size"],
    "TextGenerate": ["prompt", "max_length", "sampling_mode", "sampling_mode.temperature",
                     "sampling_mode.top_k", "sampling_mode.top_p", "sampling_mode.min_p",
                     "sampling_mode.repetition_penalty", "sampling_mode.seed",
                     "sampling_mode.presence_penalty", "thinking", "use_default_template", "mtp"],
    "PreviewAny": [], "VAEDecode": [], "SamplerCustomAdvanced": [],
    "ComfySwitchNode": ["switch"], "PrimitiveStringMultiline": ["value"],
    "ManualSigmas": ["sigmas"], "KSamplerSelect": ["sampler_name"],
    "RandomNoise": ["noise_seed"], "CFGGuider": ["cfg"],
    "QwenImage21Cache": ["device", "dtype"],
    "SaveImageAdvanced": ["filename_prefix", "format", "format.bit_depth", "format.input_color_space"],
    "CheckpointLoaderSimple": ["ckpt_name"], "CLIPTextEncode": ["text"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    "SaveImage": ["filename_prefix"],
    "KSampler": ["seed", "_frontend_seed_control", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
}


def _qwen_graph(profile: Profile, workflow_id: str) -> dict:
    workflow = profile.workflows[workflow_id]
    if not workflow.file:
        raise GenerationError(400, "template_mismatch", "A vendored Qwen workflow is required")
    source = json.loads((profile.path / workflow.file).read_text(encoding="utf-8"))
    wrapper = next(n for n in source["nodes"] if n["id"] == 459)
    subgraph = next(g for g in source["definitions"]["subgraphs"] if g["id"] == wrapper["type"])
    exposed = dict(enumerate(wrapper["widgets_values"]))
    if len(exposed) != len(subgraph["inputs"]):
        raise GenerationError(400, "template_mismatch", "Qwen subgraph input contract changed")
    graph = {}
    links = {link["id"]: link for link in subgraph["links"]}
    for node in subgraph["nodes"]:
        kind = node["type"]
        if kind not in _WIDGETS or node.get("mode", 0) != 0:
            raise GenerationError(400, "unsupported_template_node", "Unsupported Qwen template node: " + kind)
        values = node.get("widgets_values") or []
        inputs = dict(zip(_WIDGETS[kind], values))
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                link = links[inp["link"]]
                if link["origin_id"] == -10:
                    inputs[inp["name"]] = exposed[link["origin_slot"]]
                else:
                    inputs[inp["name"]] = [str(link["origin_id"]), link["origin_slot"]]
        graph[str(node["id"])] = {"class_type": kind, "inputs": inputs}
    save = next(n for n in source["nodes"] if n["id"] == 461)
    output_link = next(link for link in subgraph["links"] if link["target_id"] == -20 and link["target_slot"] == 0)
    graph["461"] = {"class_type": save["type"], "inputs": {
        **dict(zip(_WIDGETS[save["type"]], save["widgets_values"])),
        "images": [str(output_link["origin_id"]), output_link["origin_slot"]],
    }}
    return graph


def _sd_graph(profile: Profile, workflow_id: str) -> dict:
    workflow = profile.workflows[workflow_id]
    if not workflow.file:
        raise GenerationError(400, "template_mismatch", "A vendored SD workflow is required")
    source = json.loads((profile.path / workflow.file).read_text(encoding="utf-8"))
    links = {link[0]: link for link in source["links"]}
    graph = {}
    for node in source["nodes"]:
        kind = node["type"]
        if kind == "MarkdownNote":
            continue
        if kind not in _WIDGETS or node.get("mode", 0) != 0:
            raise GenerationError(400, "unsupported_template_node", "Unsupported SD template node: " + kind)
        inputs = {key: value for key, value in zip(_WIDGETS[kind], node.get("widgets_values") or [])
                  if key != "_frontend_seed_control"}
        for inp in node.get("inputs", []):
            if inp.get("link") is not None:
                link = links[inp["link"]]
                inputs[inp["name"]] = [str(link[1]), link[2]]
        graph[str(node["id"])] = {"class_type": kind, "inputs": inputs}
    return graph


def _render_template(adapter: Callable, profile: Profile, workflow_id: str) -> dict:
    try:
        return adapter(profile, workflow_id)
    except GenerationError:
        raise
    except (OSError, KeyError, IndexError, TypeError, ValueError, StopIteration) as exc:
        raise GenerationError(400, "template_mismatch",
                              "The vendored template no longer matches this generation adapter") from exc


def build_generation(profile: Profile, body: dict) -> Generation:
    cap = capabilities(profile)
    if not cap["available"]:
        raise GenerationError(400, "unsupported_profile", cap["reason"])
    allowed = {"prompt", "profile", "workflow", "preset", "negative_prompt", "seed", "width", "height",
               "reference_image", "denoise", "auto_wake", "timeout_seconds"}
    if set(body) - allowed:
        raise GenerationError(400, "unsupported_parameter", "Unsupported parameters: " + ", ".join(sorted(set(body) - allowed)))
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise GenerationError(400, "missing_prompt", "Field 'prompt' must be a nonempty string")
    if body.get("workflow", cap["workflow_id"]) != cap["workflow_id"]:
        raise GenerationError(400, "unsupported_workflow", "This quick entry supports only " + cap["workflow_id"])
    width = _integer(body, "width", 1024, 256, 2048)
    height = _integer(body, "height", 1024, 256, 2048)
    multiple = cap["dimension_multiple"]
    if width % multiple or height % multiple:
        raise GenerationError(400, "invalid_dimensions", f"width and height must be multiples of {multiple}")
    seed = _integer(body, "seed", -1, -1, 2**64 - 1)
    if seed == -1:
        seed = secrets.randbits(32)
    timeout = _integer(body, "timeout_seconds", 300, 15, 1800)
    if "auto_wake" in body and type(body["auto_wake"]) is not bool:
        raise GenerationError(400, "invalid_parameter", "auto_wake must be a boolean")
    negative = body.get("negative_prompt", cap["default_negative"])
    if not isinstance(negative, str):
        raise GenerationError(400, "invalid_parameter", "negative_prompt must be a string")
    if not cap["negative_prompt"] and negative:
        raise GenerationError(400, "unsupported_parameter", "This workflow uses CFG=1 and does not support a negative prompt")
    preset = body.get("preset", cap["default_preset"])
    if not isinstance(preset, str):
        raise GenerationError(400, "invalid_parameter", "preset must be a string")
    reference = body.get("reference_image", "")
    if not isinstance(reference, str):
        raise GenerationError(400, "invalid_parameter", "reference_image must be a filename")
    if not cap["reference_image"] and (reference or "denoise" in body):
        raise GenerationError(400, "unsupported_parameter", "Reference-image editing is not connected for this quick workflow")
    if "denoise" in body and not reference:
        raise GenerationError(400, "unsupported_parameter", "denoise requires an uploaded reference_image")
    if profile.id == "sd-latest":
        if preset != cap["default_preset"]:
            raise GenerationError(400, "unsupported_preset", "Unknown SD 3.5 sampling preset")
        graph = _render_template(_sd_graph, profile, cap["workflow_id"])
        graph["16"]["inputs"]["text"] = prompt
        graph["40"]["inputs"]["text"] = negative
        graph["3"]["inputs"]["seed"] = seed
        graph["53"]["inputs"].update(width=width, height=height)
        return Generation(graph, profile.id, cap["workflow_id"], preset,
                          {"seed": seed, "width": width, "height": height, "steps": 20}, timeout)
    if profile.id == "qwen-image":
        if preset != cap["default_preset"]:
            raise GenerationError(400, "unsupported_preset", "Unknown Qwen Turbo sampling preset")
        graph = _render_template(_qwen_graph, profile, cap["workflow_id"])
        graph["474"]["inputs"]["on_false"] = prompt
        graph["471"]["inputs"]["prompt"] = prompt
        graph["474"]["inputs"]["switch"] = False
        graph["471"]["inputs"]["sampling_mode.seed"] = seed
        graph["488"]["inputs"]["noise_seed"] = seed
        graph["456"]["inputs"].update(width=width, height=height)
        return Generation(graph, profile.id, cap["workflow_id"], preset,
                          {"seed": seed, "width": width, "height": height, "steps": 8}, timeout)
    idx = MOSS_ALIASES.get(preset.strip().lower())
    if idx is None:
        idx = next((i for i, name in enumerate(PRESETS) if name.lower() == preset.lower()), None)
    if idx is None:
        raise GenerationError(400, "unsupported_preset", "Unknown MOSS sampling preset")
    preset = PRESETS[idx]
    graph = make_moss_txt2img_graph(prompt, negative, preset, seed, width, height)
    graph["8"]["inputs"]["text"] = negative  # An explicitly empty negative prompt is not a default request.
    denoise = body.get("denoise", 0.65)
    if isinstance(denoise, bool) or not isinstance(denoise, (int, float)) or not 0.05 <= denoise <= 1:
        raise GenerationError(400, "invalid_parameter", "denoise must be between 0.05 and 1")
    if reference:
        if "/" in reference or "\\" in reference or reference.startswith(".") or "\0" in reference:
            raise GenerationError(400, "invalid_reference", "reference_image must be an uploaded image filename")
        _reference(graph, reference, denoise)
    params: dict[str, Any] = {"seed": seed, "width": width, "height": height, "steps": [25, 25, 14, 10, 10, 25][idx]}
    if reference:
        params.update(reference_image=reference, denoise=denoise)
    return Generation(graph, profile.id, cap["workflow_id"], preset, params, timeout)
