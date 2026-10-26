"""comfy-gate HTTP server (aiohttp) — Compliant Headless Batch Edition."""
from __future__ import annotations

import asyncio
import hmac
import html as html_lib
import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path

from aiohttp import web

from . import __version__
from .auth import IDENT, AuthError, Identity, JWKSCache, authorize, identify
from .batch_dispatcher import BatchDispatcher, BatchError
from .colab import ColabCLI, ColabError, ColabRunner
from .config import Settings, read_secret_file
from .db import DB_KEY, GateDB
from .device_auth import (
    create_agent_token_endpoint,
    device_approve_endpoint,
    device_code_endpoint,
    device_deny_endpoint,
    device_page_endpoint,
    device_token_endpoint,
    list_agent_tokens_endpoint,
    revoke_agent_token_endpoint,
)
from .profiles import Profile, load_all
from .session import REPO_ROOT, Session
from .state import State

log = logging.getLogger("gate")


def _find_web_dir() -> Path:
    app_web = Path("/app/apps/gate/web")
    if app_web.is_dir() and (app_web / "index.html").is_file():
        return app_web
    repo_web = REPO_ROOT / "apps" / "gate" / "web"
    if repo_web.is_dir() and (repo_web / "index.html").is_file():
        return repo_web
    pkg_web = Path(__file__).parent / "web"
    if pkg_web.is_dir() and (pkg_web / "index.html").is_file():
        return pkg_web
    return Path(__file__).parent / "web"


WEB = _find_web_dir()
PUBLIC_PATHS = ("/_gate/healthz", "/robots.txt", "/api/auth/device/code", "/api/auth/device/token")
BOT_UA_RE = re.compile(
    r"(bot|crawler|spider|crawling|slurp|archiver|scraper|yandex|bingbot|googlebot|baiduspider|bytespider|twitterbot|facebookexternalhit|rogerbot|linkedinbot|embedly|quora link preview|showyoubot|outbrain|pinterest|slackbot|vkShare|W3C_Validator)",
    re.IGNORECASE,
)
SESSION = web.AppKey("session", Session)
SETTINGS = web.AppKey("settings", Settings)
BATCH = web.AppKey("batch", BatchDispatcher)
PROFILES = web.AppKey("profiles", dict)
PROFILE_ERRORS = web.AppKey("profile_errors", dict)
JWKS = web.AppKey("jwks", object)
AGENT_TOKEN = web.AppKey("agent_token", str)
BG = web.AppKey("bg", object)


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310
        return json.loads(r.read())


def _load_moss_builder():
    try:
        from ckit.moss_template import PRESETS, make_moss_txt2img_graph
    except ImportError:
        import sys

        pkg_dir = REPO_ROOT / "packages"
        if str(pkg_dir) not in sys.path:
            sys.path.insert(0, str(pkg_dir))
        from cli.moss_template import PRESETS, make_moss_txt2img_graph
    return make_moss_txt2img_graph, PRESETS


def _inject_reference_image(graph: dict, ref_image: str, denoise: float = 0.65) -> dict:
    """Optionally adapt a txt2img API graph or LoadImage node to use an uploaded reference image."""
    if not ref_image:
        return graph
    # 1. If the graph already contains a LoadImage node, bind it directly.
    for node in graph.values():
        if isinstance(node, dict) and node.get("class_type") == "LoadImage":
            node.setdefault("inputs", {})["image"] = ref_image
            for s_node in graph.values():
                if isinstance(s_node, dict) and "denoise" in (s_node.get("inputs") or {}):
                    s_node["inputs"]["denoise"] = max(0.05, min(1.0, float(denoise)))
            return graph
    # 2. If using the MOSS template (node "23" EmptyFlux2LatentImage + node "5" VAELoader), replace latent source with LoadImage -> VAEEncode.
    if "23" in graph and "5" in graph:
        graph["90"] = {"class_type": "LoadImage", "inputs": {"image": ref_image}}
        graph["91"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["90", 0], "vae": ["5", 0]}}
        for nid, node in list(graph.items()):
            if nid in ("90", "91") or not isinstance(node, dict):
                continue
            inputs = node.get("inputs") or {}
            for k, v in list(inputs.items()):
                if v == ["23", 0]:
                    inputs[k] = ["91", 0]
            if "denoise" in inputs:
                inputs["denoise"] = max(0.05, min(1.0, float(denoise)))
        graph.pop("23", None)
    return graph


def err(status: int, code: str, msg: str) -> web.Response:
    return web.json_response({"error": code, "message": msg}, status=status)


@web.middleware
async def anti_crawler_mw(request: web.Request, handler):
    if request.path == "/robots.txt":
        return await handler(request)
    ua = request.headers.get("User-Agent", "")
    if BOT_UA_RE.search(ua):
        return web.Response(
            status=403,
            text="Access forbidden for automated bots, crawlers, and scrapers.",
            content_type="text/plain",
            headers={"X-Robots-Tag": "noindex, nofollow, noarchive, nosnippet"},
        )
    resp = await handler(request)
    resp.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
    return resp


@web.middleware
async def auth_mw(request: web.Request, handler):
    if (
        request.path in PUBLIC_PATHS
        or request.path.startswith("/_gate/static/")
        or (request.method == "GET" and request.path in ("/mcp", "/_gate/mcp"))
    ):
        return await handler(request)
    app = request.app
    query_token = request.query.get("token", "")
    cookie_token = request.cookies.get("agent_token", "")
    cf_cookie = request.cookies.get("CF_Authorization", "")
    try:
        ident = await asyncio.get_running_loop().run_in_executor(
            None,
            identify,
            request.headers,
            app[SETTINGS],
            app[JWKS],
            app[AGENT_TOKEN],
            query_token,
            cookie_token,
            app.get(DB_KEY),
            cf_cookie,
        )
    except AuthError as e:
        accept = request.headers.get("Accept", "")
        if "text/html" in accept and not request.path.startswith("/_gate/") and not request.path.startswith("/api/"):
            safe_msg = html_lib.escape(e.msg, quote=True)
            html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>访问受限 · Comfy Colab Kit</title>
<style>
:root {{
  --bg: #f8fafc; --card: #ffffff; --border: #e2e8f0;
  --text: #0f172a; --muted: #64748b; --primary: #2563eb; --danger: #dc2626;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0b0f19; --card: #111827; --border: #1f2937;
    --text: #f3f4f6; --muted: #9ca3af; --primary: #3b82f6; --danger: #f87171;
  }}
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  min-height: 100dvh; display: flex; align-items: center; justify-content: center; padding: 1.5rem;
}}
.card {{
  background: var(--card); border: 1px solid var(--border); border-radius: 14px;
  width: 100%; max-width: 420px; padding: 1.75rem; text-align: center;
}}
h1 {{ font-size: 1.25rem; font-weight: 700; margin-bottom: 0.65rem; }}
p {{ font-size: 0.88rem; color: var(--muted); line-height: 1.5; margin-bottom: 1.1rem; }}
.code-box {{
  border: 1px solid var(--border); border-radius: 8px; padding: 0.7rem 0.9rem;
  font-family: ui-monospace, Menlo, monospace; font-size: 0.82rem; color: var(--danger);
  margin-bottom: 1.25rem; word-break: break-all;
}}
.btn {{
  display: inline-block; padding: 0.65rem 1.3rem; border-radius: 8px;
  background: var(--primary); color: #fff; text-decoration: none; font-size: 0.88rem; font-weight: 600;
}}
</style>
</head>
<body>
<div class="card">
  <h1>访问受限 (Access Denied)</h1>
  <p>当前请求未通过访问控制校验。</p>
  <div class="code-box">{safe_msg}</div>
  <a class="btn" href="/cdn-cgi/access/logout">重新登录</a>
</div>
</body>
</html>"""
            return web.Response(text=html, content_type="text/html", status=e.status)
        return err(e.status, e.code, e.msg)
    request[IDENT] = ident
    resp = await handler(request)
    if query_token and app[AGENT_TOKEN] and hmac.compare_digest(query_token, app[AGENT_TOKEN]):
        resp.set_cookie("agent_token", query_token, path="/", httponly=True, samesite="Lax")
    return resp


def _ident(request: web.Request) -> Identity:
    return request[IDENT]


def get_client_ip(request: web.Request) -> str:
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        client = xff.split(",")[0].strip()
        if client:
            return client
    if request.remote:
        return request.remote.strip()
    return ""


async def _body(request: web.Request) -> dict:
    if not request.can_read_body:
        return {}
    try:
        d = await request.json()
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


async def status(request: web.Request) -> web.Response:
    s = request.app[SESSION]
    ident = _ident(request)
    client_ip = get_client_ip(request)
    can_stop = s.can_stop(email=ident.email, ip=client_ip, is_owner=ident.is_owner)
    can_wake = "wake" not in request.app[SETTINGS].owner_only_actions or ident.is_owner
    st = s.status()
    if not ident.is_owner:
        st["cu"] = None
        st["woken_by"] = None
    return web.json_response({
        **st,
        "can_stop": can_stop,
        "can_wake": can_wake,
        "you": {
            "email": ident.email,
            "kind": ident.kind,
            "owner": ident.is_owner,
            "ip": client_ip,
            "can_stop": can_stop,
            "can_wake": can_wake,
        },
        "owner_only": list(request.app[SETTINGS].owner_only_actions),
        "version": __version__,
    })


def _profile_catalog(request: web.Request) -> dict:
    from .generation import capabilities
    _, presets = _load_moss_builder()
    return {
        "default": request.app[SETTINGS].default_profile,
        "profiles": [{**p.summary(), "generation": capabilities(p)} for p in request.app[PROFILES].values()],
        "presets": list(presets),
        "errors": request.app[PROFILE_ERRORS],
    }


async def profiles_route(request: web.Request) -> web.Response:
    return web.json_response(_profile_catalog(request))


async def wake(request: web.Request) -> web.Response:
    ident = _ident(request)
    try:
        authorize(ident, "wake", request.app[SETTINGS])
    except AuthError as e:
        return err(e.status, e.code, e.msg)
    body = await _body(request)
    s = request.app[SESSION]
    profile = body.get("profile")
    client_ip = get_client_ip(request)
    try:
        res = await s.wake(profile, by=ident.email, ip=client_ip, workflows=body.get("workflows"))
    except KeyError:
        return err(404, "unknown_profile", f"unknown profile {profile!r}")
    if res.refused:
        return err(409, res.refused, f"wake refused: {res.refused}")
    return web.json_response(
        {"joined": res.joined, "state": res.state, "profile": res.profile},
        status=200 if res.joined else 202,
    )


async def fetch(request: web.Request) -> web.Response:
    ident = _ident(request)
    try:
        authorize(ident, "wake", request.app[SETTINGS])
    except AuthError as e:
        return err(e.status, e.code, e.msg)
    body = await _body(request)
    wids = body.get("workflows")
    if not isinstance(wids, list) or not all(isinstance(w, str) for w in wids) or not wids:
        return err(400, "bad_request", 'body must be {"workflows": ["<id>", ...]}')
    why = await request.app[SESSION].fetch(wids)
    if why:
        return err(404 if why == "unknown_workflow" else 409, why, f"fetch refused: {why}")
    return web.json_response({"fetching": wids}, status=202)


async def switch(request: web.Request) -> web.Response:
    ident = _ident(request)
    try:
        authorize(ident, "switch", request.app[SETTINGS])
    except AuthError as e:
        return err(e.status, e.code, e.msg)
    body = await _body(request)
    client_ip = get_client_ip(request)
    try:
        res = await request.app[SESSION].switch(str(body.get("profile", "")), by=ident.email, ip=client_ip)
    except KeyError:
        return err(404, "unknown_profile", "unknown profile")
    if res.refused:
        return err(409, res.refused, f"switch refused: {res.refused}")
    return web.json_response({"state": res.state, "profile": res.profile, "mode": "cold"}, status=202)


async def stop(request: web.Request) -> web.Response:
    ident = _ident(request)
    s = request.app[SESSION]
    client_ip = get_client_ip(request)
    if not s.can_stop(email=ident.email, ip=client_ip, is_owner=ident.is_owner):
        return err(403, "not_initiator", "Only the user who started the session (or owner) can stop it.")
    await s.stop(f"stopped by {ident.email or client_ip or 'unknown'}")
    return web.json_response({"state": request.app[SESSION].machine.state.value})


async def extend(request: web.Request) -> web.Response:
    ident = _ident(request)
    try:
        authorize(ident, "extend", request.app[SETTINGS])
    except AuthError as e:
        return err(e.status, e.code, e.msg)
    if not await request.app[SESSION].extend():
        return err(409, "extend_limit", "no extensions left")
    return web.json_response(request.app[SESSION].status())


# ── Batch Execution & ComfyUI-Compatible Endpoints ───────────────────────────
async def upload_image_route(request: web.Request) -> web.Response:
    batch = request.app[BATCH]
    try:
        reader = await request.multipart()
    except Exception:
        return err(400, "invalid_multipart", "Expected multipart/form-data with field 'image'")
    filename = "upload.png"
    img_bytes = b""
    overwrite = True
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "image":
            filename = part.filename or filename
            img_bytes = await part.read(decode=False)
        elif part.name == "overwrite":
            val = (await part.text()).strip().lower()
            overwrite = val not in ("false", "0", "no")
    try:
        info = batch.save_input_image(filename, img_bytes, overwrite=overwrite)
    except BatchError as be:
        return err(be.status, be.code, be.msg)
    return web.json_response(info)


async def prompt_route(request: web.Request) -> web.Response:
    ident = _ident(request)
    body = await _body(request)
    prompt_graph = body.get("prompt") if isinstance(body.get("prompt"), dict) else body
    auto_wake = bool(body.get("auto_wake", False))
    s = request.app[SESSION]
    if auto_wake and s.machine.state in (State.SLEEPING, State.FAILED):
        try:
            authorize(ident, "wake", request.app[SETTINGS])
        except AuthError as e:
            return err(e.status, e.code, e.msg)
        res = await s.wake(body.get("profile"), by=ident.email, ip=get_client_ip(request))
        if res.refused:
            return err(409, res.refused, f"wake refused: {res.refused}")
    try:
        job = request.app[BATCH].enqueue(
            prompt=prompt_graph,
            submitted_by=ident.email,
            title=str(body.get("title") or "API Workflow"),
            profile_id=body.get("profile") or s.machine.profile or "",
            timeout_s=int(body.get("timeout_seconds") or 300),
            allow_booting=auto_wake or s.machine.is_booting,
        )
    except BatchError as be:
        return err(be.status, be.code, be.msg)
    return web.json_response({"prompt_id": job.prompt_id, "number": job.number, "node_errors": {}})


async def _enqueue_generation(request: web.Request, body: dict):
    ident = _ident(request)
    s = request.app[SESSION]
    from .generation import GenerationError, build_generation

    pid = body.get("profile") or s.machine.profile or request.app[SETTINGS].default_profile
    if not isinstance(pid, str) or pid not in request.app[PROFILES]:
        raise GenerationError(404, "unknown_profile", "Unknown profile")
    generation = build_generation(request.app[PROFILES][pid], body)
    if s.machine.state != State.SLEEPING and s.machine.profile != pid:
        raise GenerationError(409, "profile_mismatch", "The active GPU uses a different profile; switch explicitly when idle")
    if generation.parameters.get("reference_image"):
        request.app[BATCH].resolve_view_file(generation.parameters["reference_image"], "input")
    auto_wake = body.get("auto_wake", False)
    if auto_wake and s.machine.state in (State.SLEEPING, State.FAILED):
        authorize(ident, "wake", request.app[SETTINGS])
        try:
            res = await s.wake(pid, by=ident.email, ip=get_client_ip(request))
        except KeyError:
            raise GenerationError(404, "unknown_profile", "Unknown profile") from None
        if res.refused:
            raise GenerationError(409, res.refused, f"wake refused: {res.refused}")

    return request.app[BATCH].enqueue(
            prompt=generation.graph,
            submitted_by=ident.email,
            title=body["prompt"][:80],
            preset=generation.preset,
            profile_id=generation.profile_id,
            workflow_id=generation.workflow_id,
            parameters=generation.parameters,
            timeout_s=generation.timeout_s,
            allow_booting=auto_wake or s.machine.is_booting,
    )


async def txt2img_route(request: web.Request) -> web.Response:
    from .generation import GenerationError
    try:
        job = await _enqueue_generation(request, await _body(request))
    except (AuthError, BatchError, GenerationError) as e:
        return err(e.status, e.code, e.msg)
    return web.json_response({"prompt_id": job.prompt_id, "number": job.number, "job": job.to_summary()}, status=202)


async def jobs_route(request: web.Request) -> web.Response:
    limit = int(request.query.get("limit", "40") or "40")
    return web.json_response({"jobs": request.app[BATCH].list_jobs(limit=limit)})


async def history_route(request: web.Request) -> web.Response:
    pid = request.match_info.get("prompt_id")
    return web.json_response(request.app[BATCH].history_dict(pid))


async def view_route(request: web.Request) -> web.StreamResponse:
    filename = request.query.get("filename", "")
    file_type = request.query.get("type", "output")
    subfolder = request.query.get("subfolder", "")
    try:
        target = request.app[BATCH].resolve_view_file(filename, file_type=file_type, subfolder=subfolder)
    except BatchError as be:
        return err(be.status, be.code, be.msg)
    return web.FileResponse(target, headers={"Cache-Control": "private, max-age=3600"})


async def queue_route(request: web.Request) -> web.Response:
    batch = request.app[BATCH]
    if request.method == "POST":
        ident = _ident(request)
        s = request.app[SESSION]
        if not ident.is_owner and not s.can_stop(email=ident.email, ip=get_client_ip(request), is_owner=ident.is_owner):
            return err(403, "forbidden", "Only the session initiator or owner can clear the queue")
        body = await _body(request)
        cleared = batch.clear_pending() if body.get("clear") else 0
        return web.json_response({"ok": True, "cleared": cleared})
    return web.json_response(batch.queue_dict())


async def interrupt_route(request: web.Request) -> web.Response:
    ident = _ident(request)
    s = request.app[SESSION]
    if not ident.is_owner and not s.can_stop(email=ident.email, ip=get_client_ip(request), is_owner=ident.is_owner):
        return err(403, "forbidden", "Only the session initiator or owner can interrupt jobs")
    cleared = request.app[BATCH].clear_pending()
    return web.json_response({"ok": True, "cleared": cleared})


# ── Model Context Protocol (HTTP Streamable JSON-RPC 2.0) ────────────────────
async def mcp_endpoint(request: web.Request) -> web.Response:
    from ckit.mcp_server import TOOLS
    origin = request.headers.get("Origin")
    if origin and origin != f"https://{request.app[SETTINGS].public_host}":
        return err(403, "invalid_origin", "Cross-origin MCP requests are not allowed")
    if request.method == "GET":
        if "text/event-stream" in request.headers.get("Accept", ""):
            raise web.HTTPMethodNotAllowed("GET", ["POST"])
        return web.json_response({
            "name": "comfy-colab-kit",
            "version": __version__,
            "mode": "batch",
            "protocol": "2025-03-26",
            "tools": ["gate_status", "gate_profiles", "gate_wake", "gate_txt2img", "gate_generate", "gate_stop"],
        })
    if request.method != "POST":
        raise web.HTTPMethodNotAllowed(request.method, ["POST"])
    body = await _body(request)
    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})
    if not isinstance(params, dict):
        return err(400, "invalid_params", "MCP params must be an object")
    s = request.app[SESSION]
    if "id" not in body and isinstance(method, str) and method.startswith("notifications/"):
        return web.Response(status=202)

    if method == "initialize":
        requested_version = params.get("protocolVersion")
        protocol_version = requested_version if requested_version in ("2025-03-26", "2025-06-18") else "2025-03-26"
        return web.json_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": protocol_version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "comfy-colab-kit", "version": __version__},
            },
        })
    if method == "tools/list":
        return web.json_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    next(t for t in TOOLS if t["name"] == "gate_profiles"),
                    {
                        "name": "gate_status",
                        "description": "Inspect the Comfy Colab batch session state, active profile, keep-warm timer, and CU balance.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "gate_wake",
                        "description": "Wake or join the shared Google Colab L4 GPU batch runtime with a model profile.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "profile": {"type": "string", "description": "Profile ID, e.g. 'moss-image'"},
                            },
                        },
                    },
                    next(t for t in TOOLS if t["name"] == "gate_txt2img"),
                    {
                        "name": "gate_generate",
                        "description": "Submit a raw ComfyUI API prompt graph to the headless batch runner and return output URLs.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "object", "description": "The ComfyUI API prompt dictionary."},
                                "timeout_seconds": {"type": "integer", "description": "Max wait seconds (default 180)."},
                            },
                            "required": ["prompt"],
                        },
                    },
                    {
                        "name": "gate_stop",
                        "description": "Stop the Colab GPU session immediately and verify server-side release.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ]
            },
        })
    if method == "tools/call":
        tool_name = params.get("name")
        ident = _ident(request)
        if tool_name == "gate_profiles":
            return web.json_response({"jsonrpc": "2.0", "id": req_id, "result": {
                "content": [{"type": "text", "text": json.dumps(_profile_catalog(request), ensure_ascii=False)}],
            }})
        if tool_name == "gate_status":
            st = s.status()
            if not ident.is_owner:
                st["cu"] = None
                st["woken_by"] = None
            return web.json_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(st, indent=2)}]},
            })
        if tool_name == "gate_wake":
            # Enforce exact same owner-only wake authorization as REST /_gate/wake
            try:
                authorize(ident, "wake", request.app[SETTINGS])
            except AuthError as e:
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"isError": True, "content": [{"type": "text", "text": f"Permission denied ({e.code}): {e.msg}"}]},
                    },
                    status=e.status,
                )
            args = params.get("arguments", {})
            prof = args.get("profile") or request.app[SETTINGS].default_profile
            try:
                res = await s.wake(prof, by=ident.email, ip=get_client_ip(request))
            except KeyError:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"isError": True, "content": [{"type": "text", "text": f"Unknown profile '{prof}'"}]},
                })
            if res.refused:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"isError": True, "content": [{"type": "text", "text": f"Wake refused: {res.refused}"}]},
                })
            return web.json_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": f"Wake initiated for profile '{prof}'. State: {res.state}"}]},
            })
        if tool_name in ("gate_txt2img", "gate_generate"):
            if s.machine.state != State.READY:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "isError": True,
                        "content": [{"type": "text", "text": f"Cannot generate: GPU is not ready (state: '{s.machine.state.value}'). Call gate_wake first."}],
                    },
                })
            from .generation import GenerationError
            args = params.get("arguments", {})
            try:
                if not isinstance(args, dict):
                    raise GenerationError(400, "invalid_parameter", "arguments must be an object")
                if tool_name == "gate_txt2img":
                    job = await _enqueue_generation(request, args)
                else:
                    job = request.app[BATCH].enqueue(
                        prompt=args.get("prompt"), submitted_by=ident.email, title="MCP Workflow",
                        profile_id=args.get("profile") or s.machine.profile or "",
                        timeout_s=int(args.get("timeout_seconds") or 180),
                    )
                timeout_s = job.timeout_s
            except (BatchError, GenerationError, AuthError) as be:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"isError": True, "content": [{"type": "text", "text": f"Prompt rejected ({be.code}): {be.msg}"}]},
                })

            try:
                await asyncio.wait_for(job.done_event.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"isError": True, "content": [{"type": "text", "text": f"Generation timed out after {timeout_s}s. Prompt ID: {job.prompt_id}"}]},
                })

            if job.status != "success":
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"isError": True, "content": [{"type": "text", "text": f"Generation failed: {job.error}"}]},
                })

            proto = request.headers.get("X-Forwarded-Proto", request.scheme)
            host = request.headers.get("X-Forwarded-Host", request.host)
            public_base = f"{proto}://{host}".rstrip("/")
            images = []
            for node_id, node_out in job.outputs.items():
                for img in node_out.get("images", []):
                    fn = img.get("filename", "")
                    url = f"{public_base}/view?filename={urllib.parse.quote(fn)}&type=output"
                    images.append({"filename": fn, "url": url, "node_id": node_id})
            res_data = {"prompt_id": job.prompt_id, "status": "success", "duration_s": job.duration_s, "images": images}
            res_text = json.dumps(res_data, indent=2, ensure_ascii=False)
            if images:
                img_md = "\n".join([f"![{img['filename']}]({img['url']})\nLink: {img['url']}" for img in images])
                res_text += f"\n\n### Generated Images\n{img_md}"
            return web.json_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": res_text}]},
            })
        if tool_name == "gate_stop":
            client_ip = get_client_ip(request)
            if not s.can_stop(email=ident.email, ip=client_ip, is_owner=ident.is_owner):
                return web.json_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"isError": True, "content": [{"type": "text", "text": "Permission denied: only the session initiator or owner can stop it"}]},
                })
            st = await s.stop("mcp agent requested stop")
            return web.json_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": f"Session stopped. State: {st.value}"}]},
            })
        return web.json_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
        })

    return web.json_response({
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not supported: {method}"},
    })


async def events(request: web.Request) -> web.StreamResponse:
    s = request.app[SESSION]
    ident = _ident(request)
    cid = (
        request.query.get("cid")
        or request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For")
        or request.remote
        or str(id(request))
    ).strip()
    try:
        since = int(request.query.get("since", 0))
    except (ValueError, TypeError):
        since = 0
    resp = web.StreamResponse(
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
    await resp.prepare(request)

    async def send(kind: str, data) -> None:
        await resp.write(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode())

    q = s.bus.subscribe()
    s.client_opened(cid)
    try:
        init_st = s.status()
        if not ident.is_owner:
            init_st["cu"] = None
            init_st["woken_by"] = None
        await send("state", init_st)
        if s.progress:
            await send("progress", s.progress)
        for line in s.bus.logs_since(since):
            await send("log", line)
        while True:
            try:
                kind, data = await asyncio.wait_for(q.get(), timeout=20)
                if kind == "state" and isinstance(data, dict) and not ident.is_owner:
                    data = {**data, "cu": None, "woken_by": None}
                await send(kind, data)
            except asyncio.TimeoutError:
                await resp.write(b": ping\n\n")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        s.bus.unsubscribe(q)
        s.client_closed(cid)
    return resp


async def healthz(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "state": request.app[SESSION].machine.state.value, "mode": "batch"})


async def robots_txt(request: web.Request) -> web.Response:
    return web.Response(
        text="User-agent: *\nDisallow: /\n",
        content_type="text/plain",
        headers={"Cache-Control": "public, max-age=86400", "X-Robots-Tag": "noindex, nofollow, noarchive, nosnippet"},
    )


async def studio_page(request: web.Request) -> web.Response:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    s = request.app[SESSION]
    ident = _ident(request)
    settings = request.app[SETTINGS]
    client_ip = get_client_ip(request)
    can_stop = s.can_stop(email=ident.email, ip=client_ip, is_owner=ident.is_owner)
    can_wake = "wake" not in settings.owner_only_actions or ident.is_owner
    st = s.status()
    if not ident.is_owner:
        st["cu"] = None
        st["woken_by"] = None
    status_data = {
        **st,
        "can_stop": can_stop,
        "can_wake": can_wake,
        "you": {
            "email": ident.email,
            "kind": ident.kind,
            "owner": ident.is_owner,
            "ip": client_ip,
            "can_stop": can_stop,
            "can_wake": can_wake,
        },
        "owner_only": list(settings.owner_only_actions),
    }
    profiles_data = _profile_catalog(request)
    init_json = json.dumps({
        "status": status_data,
        "profiles": profiles_data,
        "jobs": request.app[BATCH].list_jobs(limit=24),
        "logs": list(s.bus.logs),
    }).replace("</script>", "<\\/script>")
    tag = f'<script id="init-data">window.__INIT__ = {init_json};</script>'
    if "</head>" in html:
        html = html.replace("</head>", f"{tag}\n</head>", 1)
    else:
        html = tag + html
    return web.Response(text=html, content_type="text/html", headers={"Cache-Control": "no-store"})


async def catch_all(request: web.Request) -> web.StreamResponse:
    s = request.app[SESSION]
    if request.path in ("/", "/index.html"):
        ident = _ident(request)
        if request.app[SETTINGS].auto_stop.wake_on_visit and s.machine.state == State.SLEEPING:
            can_wake = "wake" not in request.app[SETTINGS].owner_only_actions or ident.is_owner
            if can_wake:
                await s.wake(None, by=ident.email, ip=get_client_ip(request))
        return await studio_page(request)
    api_prefixes = (
        "/api/",
        "/ws",
        "/object_info",
        "/embeddings",
        "/extensions",
        "/system_stats",
        "/view_metadata",
    )
    if any(request.path.startswith(p) for p in api_prefixes):
        return err(404, "batch_mode_only", "Interactive ComfyUI reverse proxy is disabled in compliant batch mode")
    raise web.HTTPFound("/")


async def background(app: web.Application) -> None:
    s, batch, settings = app[SESSION], app[BATCH], app[SETTINGS]
    sweep_every = max(1, settings.auto_stop.orphan_sweep_min) * 60
    last_sweep = 0.0
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(30)
        try:
            await s.tick()
            batch.prune_outputs()
            if loop.time() - last_sweep > sweep_every:
                last_sweep = loop.time()
                if s.machine.state == State.SLEEPING:
                    names = await s.colab.sessions()
                    if settings.colab_session_name in names:
                        s.bus.publish("log", "orphan VM found while sleeping; stopping it")
                        await s.colab.stop(settings.colab_session_name)
                elif s.machine.is_billing:
                    await s.reconcile()
                bal = await s.colab.ccu_info()
                if bal is not None:
                    s.ledger.balance = bal
                    s.ledger.save()
        except ColabError as e:
            log.warning("background: %s", e)
        except Exception:  # noqa: BLE001
            log.exception("background loop error")


def build_app(
    settings: Settings,
    colab: ColabRunner | None = None,
    profiles: dict[str, Profile] | None = None,
    run_background: bool = True,
) -> web.Application:
    app = web.Application(middlewares=[anti_crawler_mw, auth_mw], client_max_size=64 * 1024 * 1024)
    if profiles is None:
        profiles, errors = load_all(settings.profiles, REPO_ROOT)
    else:
        errors = {}
    app[SETTINGS] = settings
    app[PROFILES] = profiles
    app[PROFILE_ERRORS] = errors
    app[JWKS] = JWKSCache(settings.cf_access_team_domain, _fetch_json) if settings.cf_access_team_domain else None
    app[AGENT_TOKEN] = read_secret_file(settings.agent_token_file)
    session = Session(settings, profiles, colab or ColabCLI(
        settings.colab_bin, settings.colab_token_dir,
        exec_python=settings.colab_exec_python,
        connect_timeout_s=settings.colab_connect_timeout_s,
    ))
    session.restore()
    app[SESSION] = session
    batch = BatchDispatcher(settings, session)
    app[BATCH] = batch
    db_path = Path(settings.state_dir) / "gate.db"
    db = GateDB(db_path)
    app[DB_KEY] = db
    r = app.router
    r.add_get("/robots.txt", robots_txt)
    r.add_get("/_gate/healthz", healthz)
    r.add_get("/_gate/status", status)
    r.add_get("/_gate/events", events)
    r.add_get("/_gate/profiles", profiles_route)
    r.add_get("/_gate/jobs", jobs_route)
    r.add_post("/_gate/txt2img", txt2img_route)
    r.add_post("/_gate/wake", wake)
    r.add_post("/_gate/switch", switch)
    r.add_post("/_gate/fetch", fetch)
    r.add_post("/_gate/stop", stop)
    r.add_post("/_gate/extend", extend)
    # ComfyUI-compatible batch endpoints (handled locally on VPS, zero tunnel)
    r.add_post("/upload/image", upload_image_route)
    r.add_post("/prompt", prompt_route)
    r.add_get("/history", history_route)
    r.add_get("/history/{prompt_id}", history_route)
    r.add_get("/view", view_route)
    r.add_get("/queue", queue_route)
    r.add_post("/queue", queue_route)
    r.add_post("/interrupt", interrupt_route)
    # MCP & Auth
    r.add_route("*", "/_gate/mcp", mcp_endpoint)
    r.add_route("*", "/mcp", mcp_endpoint)
    r.add_post("/api/auth/device/code", device_code_endpoint)
    r.add_post("/api/auth/device/token", device_token_endpoint)
    r.add_get("/auth/device", device_page_endpoint)
    r.add_post("/api/auth/device/approve", device_approve_endpoint)
    r.add_post("/api/auth/device/deny", device_deny_endpoint)
    r.add_get("/api/agent-tokens", list_agent_tokens_endpoint)
    r.add_post("/api/agent-tokens", create_agent_token_endpoint)
    r.add_post("/api/agent-tokens/{id}/revoke", revoke_agent_token_endpoint)
    r.add_delete("/api/agent-tokens/{id}", revoke_agent_token_endpoint)
    r.add_get("/_gate/tokens", list_agent_tokens_endpoint)
    r.add_post("/_gate/tokens", create_agent_token_endpoint)
    r.add_post("/_gate/tokens/{id}/revoke", revoke_agent_token_endpoint)
    r.add_static("/_gate/static/", WEB, show_index=False)
    r.add_route("*", "/{tail:.*}", catch_all)

    async def on_startup(app: web.Application) -> None:
        batch.start()
        await session.reconcile()
        if run_background:
            app[BG] = asyncio.create_task(background(app))

    async def on_cleanup(app: web.Application) -> None:
        if BG in app:
            app[BG].cancel()
        await batch.close()
        session.persist()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    web.run_app(build_app(settings), host=settings.listen_host, port=settings.listen_port, access_log=None)


if __name__ == "__main__":
    main()
