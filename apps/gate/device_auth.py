"""RFC 8628 Device Authorization endpoints & HTML approval UI."""
from __future__ import annotations

import html as html_lib
from aiohttp import web

from .auth import IDENT, Identity
from .db import DB_KEY, GateDB


def _ident(request: web.Request) -> Identity | None:
    val = request.get(IDENT)
    if isinstance(val, Identity):
        return val
    val = request.get("identity")
    if isinstance(val, Identity):
        return val
    return None


def _require_human(ident: Identity | None) -> web.Response | None:
    """Block bearer agent tokens from minting, approving, or revoking tokens."""
    if not ident:
        return web.json_response({"error": "unauthorized"}, status=401)
    if ident.kind == "agent":
        return web.json_response(
            {
                "error": "human_auth_required",
                "message": "Agent bearer tokens cannot mint, approve, or manage tokens; sign in via browser Access.",
            },
            status=403,
        )
    return None


def _get_base_url(request: web.Request) -> str:
    proto = request.headers.get("X-Forwarded-Proto", request.scheme)
    host = request.headers.get("X-Forwarded-Host", request.headers.get("Host", request.host))
    return f"{proto}://{host}"


async def device_code_endpoint(request: web.Request) -> web.Response:
    """RFC 8628 Step 1: Device requests a verification code."""
    db: GateDB = request.app[DB_KEY]
    body = {}
    if request.can_read_body:
        try:
            body = await request.json()
        except Exception:
            body = {}
    client_name = str(body.get("client_name") or "AI Agent").strip()[:80]

    res = db.create_device_code(client_name=client_name)
    base = _get_base_url(request)
    user_code = res["user_code"]
    res["verification_uri"] = f"{base}/auth/device"
    res["verification_uri_complete"] = f"{base}/auth/device?code={user_code}"
    return web.json_response(res)


async def device_token_endpoint(request: web.Request) -> web.Response:
    """RFC 8628 Step 3: Device polls for access token."""
    db: GateDB = request.app[DB_KEY]
    body = {}
    if request.can_read_body:
        try:
            body = await request.json()
        except Exception:
            body = {}
    device_code = str(body.get("device_code") or "").strip()
    if not device_code:
        return web.json_response({"error": "invalid_request", "error_description": "missing device_code"}, status=400)

    res = db.poll_device_token(device_code)
    st = res.get("status")
    if st == "pending":
        return web.json_response({"error": "authorization_pending"}, status=400)
    if st == "expired":
        return web.json_response({"error": "expired_token"}, status=400)
    if st == "denied":
        return web.json_response({"error": "access_denied"}, status=400)
    if st == "approved":
        return web.json_response({
            "access_token": res.get("access_token"),
            "token_type": "Bearer",
            "token_id": res.get("token_id"),
            "email": res.get("email"),
        })
    return web.json_response({"error": "invalid_grant"}, status=400)


async def device_page_endpoint(request: web.Request) -> web.Response:
    """RFC 8628 Step 2: User browser visits /auth/device to approve/deny."""
    ident = _ident(request)
    denied_resp = _require_human(ident)
    if denied_resp is not None:
        return denied_resp
    assert ident is not None
    safe_email = html_lib.escape(ident.email or "unknown", quote=True)
    is_owner = ident.is_owner
    raw_code = (request.query.get("code") or "").strip().upper()[:16]
    safe_code = html_lib.escape(raw_code, quote=True)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>授权 AI Agent · Comfy Colab Kit</title>
<style>
:root {{
  --bg: #f8fafc; --card: #ffffff; --border: #e2e8f0;
  --text: #0f172a; --muted: #64748b; --primary: #2563eb; --primary-hover: #1d4ed8;
  --danger: #dc2626; --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0b0f19; --card: #111827; --border: #1f2937;
    --text: #f3f4f6; --muted: #9ca3af; --primary: #3b82f6; --primary-hover: #2563eb;
    --danger: #ef4444;
  }}
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: var(--bg); color: var(--text); font-family: var(--font);
  min-height: 100dvh; display: flex; align-items: center; justify-content: center; padding: 1.25rem;
}}
.card {{
  background: var(--card); border: 1px solid var(--border); border-radius: 14px;
  width: 100%; max-width: 420px; padding: 1.75rem;
}}
h1 {{ font-size: 1.25rem; font-weight: 700; text-align: center; margin-bottom: 0.4rem; }}
p.subtitle {{ text-align: center; font-size: 0.86rem; color: var(--muted); margin-bottom: 1.25rem; line-height: 1.45; }}
.user-box {{
  border: 1px solid var(--border); border-radius: 10px; padding: 0.75rem 0.9rem;
  display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem; gap: 8px;
}}
.user-email {{ font-size: 0.88rem; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.badge {{
  font-size: 0.72rem; font-weight: 600; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--border); color: var(--muted); flex-shrink: 0;
}}
.input-label {{ display: block; font-size: 0.78rem; font-weight: 600; color: var(--muted); margin-bottom: 0.4rem; }}
.code-input {{
  width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 10px;
  padding: 0.8rem 1rem; color: var(--text); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 1.2rem; font-weight: 700; letter-spacing: 0.14em; text-align: center; text-transform: uppercase;
}}
.scope-list {{ margin: 1.1rem 0; border-left: 2px solid var(--primary); padding-left: 0.8rem; }}
.scope-item {{ font-size: 0.82rem; color: var(--muted); margin-bottom: 0.3rem; }}
.actions {{ display: flex; gap: 0.75rem; margin-top: 1.25rem; }}
.btn {{
  flex: 1; min-height: 42px; padding: 0.65rem 1rem; border-radius: 10px; font-size: 0.9rem;
  font-weight: 600; cursor: pointer; border: 1px solid transparent;
}}
.btn-primary {{ background: var(--primary); color: #fff; }}
.btn-primary:hover {{ background: var(--primary-hover); }}
.btn-danger {{ background: transparent; color: var(--danger); border-color: var(--border); }}
.status-msg {{ margin-top: 1rem; font-size: 0.85rem; text-align: center; display: none; }}
.status-msg.error {{ color: var(--danger); display: block; }}
.status-msg.success {{ color: #10b981; display: block; }}
</style>
</head>
<body>
<div class="card">
  <h1>授权 AI Agent 连接</h1>
  <p class="subtitle">允许外部 CLI 或 MCP 智能体向此批处理网关提交生图作业。</p>
  <div class="user-box">
    <div class="user-email">{safe_email}</div>
    <div class="badge">{"管理员" if is_owner else "协作者"}</div>
  </div>
  <form id="auth-form" onsubmit="event.preventDefault(); submitAuth(true);">
    <label class="input-label" for="user-code">设备配对码</label>
    <input id="user-code" class="code-input" maxlength="9" placeholder="XXXX-XXXX" value="{safe_code}" required autocomplete="off" />
    <div class="scope-list">
      <div class="scope-item">• 查询状态与提交无头批处理生图任务</div>
      {"<div class='scope-item'>• 唤醒与停止 Colab L4 实例 (管理员)</div>" if is_owner else "<div class='scope-item'>• 受管理员独占策略限制</div>"}
    </div>
    <div class="actions">
      <button type="button" class="btn btn-danger" onclick="submitAuth(false)">拒绝</button>
      <button type="submit" class="btn btn-primary" id="btn-approve">确认授权</button>
    </div>
  </form>
  <div id="status-box" class="status-msg"></div>
</div>
<script>
async function submitAuth(approve) {{
  const code = document.getElementById("user-code").value.trim().toUpperCase();
  const statusBox = document.getElementById("status-box");
  const btn = document.getElementById("btn-approve");
  if (!code) {{
    statusBox.className = "status-msg error";
    statusBox.textContent = "请输入设备配对码";
    return;
  }}
  btn.disabled = true;
  statusBox.className = "status-msg";
  statusBox.textContent = "处理中...";
  statusBox.style.display = "block";
  try {{
    const endpoint = approve ? "/api/auth/device/approve" : "/api/auth/device/deny";
    const res = await fetch(endpoint, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ user_code: code }})
    }});
    const data = await res.json();
    if (res.ok && data.ok) {{
      statusBox.className = "status-msg success";
      statusBox.textContent = approve
        ? "授权成功，客户端将自动领取单次令牌，您可以关闭此页面。"
        : "已拒绝该设备授权请求。";
      document.getElementById("auth-form").style.display = "none";
    }} else {{
      statusBox.className = "status-msg error";
      statusBox.textContent = "操作失败: " + (data.error || "配对码无效或已过期");
      btn.disabled = false;
    }}
  }} catch (e) {{
    statusBox.className = "status-msg error";
    statusBox.textContent = "网络错误，请稍后重试";
    btn.disabled = false;
  }}
}}
</script>
</body>
</html>
"""
    return web.Response(
        text=html,
        content_type="text/html",
        headers={"Content-Security-Policy": "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'"},
    )


async def device_approve_endpoint(request: web.Request) -> web.Response:
    """Approves a device code (human browser session required)."""
    ident = _ident(request)
    denied_resp = _require_human(ident)
    if denied_resp is not None:
        return denied_resp
    assert ident is not None
    db: GateDB = request.app[DB_KEY]
    body = {}
    if request.can_read_body:
        try:
            body = await request.json()
        except Exception:
            body = {}
    user_code = str(body.get("user_code") or "").strip().upper()
    if not user_code:
        return web.json_response({"error": "missing_user_code"}, status=400)

    try:
        res = db.approve_device_code(user_code, email=ident.email)
        return web.json_response({"ok": True, "token_id": res["token_id"]})
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)


async def device_deny_endpoint(request: web.Request) -> web.Response:
    """Denies a device code (human browser session required)."""
    ident = _ident(request)
    denied_resp = _require_human(ident)
    if denied_resp is not None:
        return denied_resp
    db: GateDB = request.app[DB_KEY]
    body = {}
    if request.can_read_body:
        try:
            body = await request.json()
        except Exception:
            body = {}
    user_code = str(body.get("user_code") or "").strip().upper()
    ok = db.deny_device_code(user_code)
    return web.json_response({"ok": ok})


# ── Token Management Endpoints ───────────────────────────────────────────────
async def list_agent_tokens_endpoint(request: web.Request) -> web.Response:
    """Lists token metadata (never plaintext secrets) for current human user."""
    ident = _ident(request)
    denied_resp = _require_human(ident)
    if denied_resp is not None:
        return denied_resp
    assert ident is not None
    db: GateDB = request.app[DB_KEY]
    tokens = db.list_tokens(email=ident.email, is_owner=ident.is_owner)
    return web.json_response({
        "tokens": tokens,
        "email": ident.email,
        "is_owner": ident.is_owner,
    })


async def create_agent_token_endpoint(request: web.Request) -> web.Response:
    """Mints a fresh one-time agent token for the authenticated human user."""
    ident = _ident(request)
    denied_resp = _require_human(ident)
    if denied_resp is not None:
        return denied_resp
    assert ident is not None
    db: GateDB = request.app[DB_KEY]
    body = {}
    if request.can_read_body:
        try:
            body = await request.json()
        except Exception:
            body = {}
    name = str(body.get("name") or "AI Agent").strip()[:80]
    res = db.create_direct_token(email=ident.email, name=name)
    return web.json_response(res)


async def revoke_agent_token_endpoint(request: web.Request) -> web.Response:
    """Deletes/revokes a specific token (human browser session required)."""
    ident = _ident(request)
    denied_resp = _require_human(ident)
    if denied_resp is not None:
        return denied_resp
    assert ident is not None
    token_id = request.match_info.get("id", "").strip()
    db: GateDB = request.app[DB_KEY]
    ok = db.delete_token(token_id, email=ident.email, is_owner=ident.is_owner)
    if not ok:
        return web.json_response({"error": "token_not_found_or_forbidden"}, status=404)
    return web.json_response({"ok": True, "deleted_id": token_id, "revoked_id": token_id})
