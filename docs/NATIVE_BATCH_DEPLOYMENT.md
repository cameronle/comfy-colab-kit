# Native headless batch deployment

This maintained fork runs **offline/headless ComfyUI jobs**, not the upstream live ComfyUI web-app proxy. The old upstream screenshots, showcase site, Docker/Caddy guide and live-WebUI descriptions are historical material; they do not describe this fork's active production architecture.

## Architecture

```text
Browser / CLI / MCP
        |
Cloudflare Access + existing VPS Tunnel
        |
127.0.0.1:18088 — Python gateway managed by systemd
        |
Google official Colab CLI, always --auth oauth2
        |
Colab notebook kernel — offline PromptExecutor, upload inputs/download outputs
```

No cloudflared/ngrok tunnel, external web listener or keep-alive loop is installed in Colab. Only the ordinary VPS HTTP gateway is exposed through Cloudflare Tunnel. This reduces the remote-WebUI/tunneling risk in Colab's FAQ, but is not a guarantee of platform approval or resource availability.

## Native service layout

- `/opt/comfy-colab-kit/releases/<git-sha>/`: immutable source snapshots, root owned.
- `/opt/comfy-colab-kit/current`: active source snapshot; profiles and runner scripts are resolved here.
- `/opt/comfy-colab-kit/venv`: dedicated uv-created Python environment, not the Hermes environment.
- `/etc/comfy-colab-kit/gate.env`: root-only service settings, `0600`.
- `/var/lib/comfy-colab-kit`: private database, ledger, session metadata and output files.
- `comfy-gate.service`: runs as an unprivileged user, starts on boot and restarts on failure.

Set `HOME` to the existing Colab CLI account's home, explicitly set `COLAB_BIN`, and preserve its private `~/.config/colab-cli/` state. Protect the filesystem read-only except for the gateway state, runtime temporary directory and Colab state directory, because OAuth refresh must be able to update the local token. Do not load the full Hermes environment into the service.

Important settings:

```dotenv
PUBLIC_HOST=comfy.example.com
LISTEN_ADDR=127.0.0.1:18088
STATE_DIR=/var/lib/comfy-colab-kit
AUTH_DISABLED=false
OWNER_ONLY_CONTROL=true
WAKE_ON_VISIT=false
IDLE_TIMEOUT_MIN=10
MIN_CU_BALANCE=10
REQUIRE_CU_BALANCE=true
ENABLE_SMOKE_TEST=false
```

Configure the correct Access team, browser-application AUD, allowed email and owner email separately. Never commit real tokens, OAuth state or the production environment file.

CU budgets and stop rules are application-side controls, **not a Google billing hard cap**. Do not automatically allocate CPU/GPU for installation verification; obtain explicit task duration/CU approval for a real model run. Do not purchase or auto-top-up CU.

## Access route policy

Protect the hostname's browser pages with One-time PIN and an exact allowed-email rule. The device approval page `/auth/device` and approval/deny POST endpoints must remain human-authenticated.

Bypass Access only for the machine routes actually served by the gateway:

```text
/_gate/*
/mcp
/prompt
/history
/history/*
/view
/queue
/upload/image
/interrupt
/api/auth/device/code
/api/auth/device/token
```

"Bypass" here means bypass the edge login page, **not bypass application authorization**. The origin validates an application Bearer token or a correctly signed Access JWT/cookie on all non-public routes. Device-code/polling endpoints and liveness are intentionally public. Agent tokens must not mint/approve/manage credentials. Match the code's route matrix; do not copy unrelated `/v1/*` rules from an OpenAI-compatible API deployment.

When updating a remote-managed Tunnel, compare the current configuration against the backed-up baseline, add only the new hostname before its final catch-all, and preserve all unrelated settings. Create the proxied DNS CNAME separately and read back both objects. A normal remote configuration push needs no cloudflared restart.

## Local client and Codex

Install a reviewed commit of this fork with `uv tool install 'git+https://github.com/cameronle/comfy-colab-kit.git@<reviewed-sha>'` (replace the placeholder with the verified deployment commit), then pair on the **client computer**:

```bash
# POSIX shells: keep newly saved local credentials private.
umask 077
ckit auth https://comfy.example.com
ckit status --json
codex mcp add comfy -- ckit mcp
```

Open the displayed pairing link, sign in using the allowed email and approve the device yourself. `ckit` saves the endpoint-specific token to `~/.config/ckit/config.json`; never send this file or its token through chat. On Windows, use the same ckit/Codex commands from the local terminal and protect the saved file with the account's private filesystem permissions. The POSIX `umask` command is not a PowerShell command.

The stdio MCP bridge reads the ckit config and calls the real HTTPS gateway; no token needs to be put in the Codex command or config. `codex mcp list` should show the entry. Tools are `gate_profiles`, `gate_status`, `gate_wake`, `gate_txt2img`, `gate_generate`, and `gate_stop`.

Alternatively, use native Streamable HTTP in `~/.codex/config.toml`:

```toml
[mcp_servers.comfy]
url = "https://comfy.example.com/mcp"
bearer_token_env_var = "CKIT_TOKEN"
startup_timeout_sec = 30
tool_timeout_sec = 3600
```

Load `CKIT_TOKEN` from your private ckit endpoint configuration into the environment of the Codex process. Do not inline it into TOML, process arguments, repository files or messages. Configure only one entry named `comfy`—choose stdio or HTTP, not both. Long tool timeouts accommodate cold starts but do not authorize spending.

## Profile-aware quick generation

Web, `ckit generate` and both MCP transports submit high-level parameters to the
same server-side adapter. `GET /_gate/profiles`, `ckit profiles --json` and MCP
`gate_profiles` expose capabilities without allocating compute. Look at each
profile's **generation** object, not the global legacy MOSS preset list.

- `moss-image` → `matlowai-presets`: six DeBand presets; negative prompts and
  uploaded reference-image editing are supported.
- `qwen-image` → `qwen21-turbo-t2i`: official 8-step Turbo conditioning/cache and
  sigma schedule. Automatic prompt rewriting is disabled to preserve input text.
  CFG=1 means no nonempty negative prompt; reference editing is not connected.
  The weights use the Qwen Research License (non-commercial research/evaluation;
  commercial use requires separate permission).
- `sd-latest` → `sd35-large`: SD 3.5 Large FP8 checkpoint, official 20-step
  sampler/conditioning; supports negative prompts, not quick reference editing.
  Stability AI Community License conditions apply, including commercial-revenue limits.

Other profile workflows remain raw-graph/optional-fetch capabilities, not promises
that `gate_txt2img` can run every listed workflow. Unsupported presets, workflows
and parameters return explicit validation errors. Width/height must be 256–2048
and multiples of the advertised `dimension_multiple` (Qwen 32, MOSS/SD 16).
Prompts are never trimmed, shortened or rewritten by the quick adapter.

The following commands may allocate/use compute; run them **only for an approved
GPU task**, not as installation probes:

```bash
ckit wake --profile qwen-image --wait
ckit generate --profile qwen-image --prompt 'The complete original prompt' --seed 42
```

`ckit generate` and MCP generation require an already-ready matching runtime.
Web quick generation may explicitly auto-wake the selected model. A request for
another profile while a runtime is active returns `409 profile_mismatch`; it does
not stop the VM, download another model set or rebuild the in-kernel engine.
Explicit switching is refused with `queue_busy` while work is queued/running.
Jobs record their profile, workflow and resolved parameters; dispatch and the
headless runner recheck profile identity before remote execution/engine creation.

Local unit/fake-runtime and Chromium tests validate routing, graph construction,
parameter fidelity and UI behavior. They do **not** establish Qwen/SD GPU model
compatibility, VRAM fit, image quality or throughput; those need separately
approved real model runs with time/CU budgets.

## Batch cold-start and failure handling

- Setup creates the task directory before both fresh setup and the reused-runtime shortcut. Dispatch also idempotently prepares task/input parents before upload; the official CLI Contents API cannot create missing parent directories.
- Directory preparation uses the streamed CLI path and requires `KIT_DIRS_READY` plus a clean connection close. A transient connection failure or local execution deadline retries **only this idempotent mkdir**, at most once; a missing confirmation or Python error is not retried. Exhausted connection failures report the pre-upload stage and release the runtime/cancel pending work. Actual image graphs are never replayed automatically, including on execution timeouts whose remote completion is unknown.
- Reference-input discovery intersects workflow strings with image names already present in local input storage before probing paths; prompt text is never passed blindly to filesystem stat calls. This avoids UTF-8 `NAME_MAX` failures without truncating prompts. Matched paths must remain inside input storage; existing mtime/session upload caching is preserved.
- The headless adapter applies the profile's ComfyUI flags before importing model/execution modules, supplies the cache headroom parameters required by the pinned ComfyUI revision, and bridges sampling progress without starting ComfyUI's HTTP server or encoding preview images.
- Runtime proxy credentials in HTTP-error query strings are sanitized before truncation, logging, SSE, job persistence and history/summary responses. Restored historical errors/logs are also sanitized before serving them.
- CLI command cancellation, stream closure and timeouts kill and reap the local child process. Stream execution has a local deadline in addition to the notebook timeout. Killing the local CLI is **not** proof that the remote GPU stopped; use the normal stop-and-read-back procedure.
- A `KIT_READY` stdout marker is not the end of the setup connection. Wait for the setup exec to return successfully and close its socket before advertising `ready` or dispatching another exec to the same kernel.
- Fatal kernel-connection tracebacks are detected immediately, with the original sanitized cause preserved; CLI 0.7.5 may otherwise stay alive on Jupyter threads after its preliminary chdir fails outside its cleanup block. Release the runtime and cancel queued work on an ambiguous transport failure; never automatically replay a graph whose completion is unknown.
- The normal session stop path cancels and awaits local batch execution and clears pending jobs before releasing the VM. Cancelled jobs are finalized and no longer leave the UI stuck busy.
- A task with no image artifacts or an empty/missing download is an error, not a successful generation. Include an image-saving output node in image workflows.
- Regression tests use local filesystem-backed upload semantics, controlled executor contracts and real **local** child processes. They never allocate a Colab environment and do not establish GPU inference success. The first post-fix GPU run still requires an explicitly approved time/CU budget.

## Verification and operating boundary

Verify separately:

1. systemd is active/enabled; port `18088` listens only on loopback.
2. local and public liveness return JSON, unauthorized machine calls return JSON `401`.
3. browser pages redirect to the correct Access login; device approval does not accept agent tokens.
4. authenticated status/profile/queue calls work; traversal and cross-origin requests fail.
5. an official MCP SDK can initialize, list all six tools and call `gate_status`/`gate_profiles` through real HTTPS; stdio notifications must not produce responses.
6. device-code creation and polling work without returning Access HTML; before human approval, polling must stay `authorization_pending`.
7. Colab usage/sessions still show no allocation after installation checks.
8. the first human OTP login and real GPU-generated output are separate acceptance steps, not established by the checks above.

An asleep gateway is a healthy idle state. Do not call `gate_wake`/`ckit wake` just to prove deployment readiness. Before an approved model run, confirm hardware, model storage requirements, time/CU budget and cleanup; download/validate the real result, stop only the task-owned environment and read back sessions to confirm release.

For service diagnosis use `systemctl status comfy-gate.service`, `journalctl -u comfy-gate.service`, and `/_gate/healthz`. Never restart Hermes, the shared Tunnel, or unrelated API services to apply a gateway-only update.
