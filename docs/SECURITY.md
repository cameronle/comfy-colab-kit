# Security & Compliance Architecture (Batch Edition)

Comfy Colab Kit operates as a **compliant headless batch dispatcher** in front of Google Colab L4 GPUs. It eliminates reverse-proxy tunnels, persistent WebUI servers inside Colab, and anti-idle keepalive loops.

---

## 1. Three-Boundary Defense-in-Depth

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. EDGE PERIMETER: Cloudflare Zero Trust Access                        │
│    • Public domain protected by mandatory One-time PIN / SSO           │
│    • Gateway cryptographically verifies RS256 JWT, AUD tag, and issuer │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Loopback HTTP (127.0.0.1:8080)
┌───────────────────────────────────▼────────────────────────────────────┐
│ 2. GATEWAY PERIMETER: comfy-gate (Native systemd, UMask 0077)          │
│    • Enforces owner-only wake/stop across BOTH REST and MCP endpoints  │
│    • Stores ONLY SHA-256 hashes of Agent Tokens in SQLite              │
│    • Blocks Agent Bearer tokens from minting or approving new tokens   │
│    • Enforces conservative 10 CU reserve floor & post-stop readback    │
│    • Serves local outputs via strict canonical path containment        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Official `colab --auth oauth2` CLI
┌───────────────────────────────────▼────────────────────────────────────┐
│ 3. COMPUTE PERIMETER: Google Colab Headless Batch Kernel               │
│    • Zero outbound tunnels (`cloudflared` removed), zero open ports    │
│    • Zero fake keepalive loops; each `exec` renders and exits cleanly  │
│    • Mandatory SHA-256 hash verification on all downloaded weights     │
│    • Optional `HF_TOKEN` sent via stdin only, scrubbed after download  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mitigated Risk Matrix

| Threat / Compliance Vector | Control |
| :--- | :--- |
| **Colab ToS Compliance** | No `cloudflared` proxy tunnel, no remote WebUI hosting, and no infinite `KIT_KEEPALIVE` anti-reaper loop. Tasks run strictly through official `colab --auth oauth2` batch commands (`new`, `upload`, `exec`, `download`, `stop`). |
| **MCP / API Privilege Escalation** | Both `POST /_gate/wake` and MCP `gate_wake` enforce `authorize(ident, "wake", settings)` under `OWNER_ONLY_CONTROL=true`. |
| **Database Credential Leakage** | `gate.db` never stores plaintext bearer tokens in `agent_tokens` or `device_codes`. Tokens are minted once in memory and stored solely as SHA-256 digests. |
| **Bearer Token Self-Persistence** | Token creation, approval, denial, and revocation endpoints require a browser-authenticated human identity (`ident.kind != "agent"`). |
| **Arbitrary File Read / Path Traversal** | Legacy `/kit/workflows/*` proxy routes are removed. Local `/view` and `/upload/image` enforce `Path.resolve().is_relative_to(base_dir.resolve())` and reject path separators or dotfiles. |
| **Model Supply-Chain Tampering** | Pure-Python `verify_sha256()` runs unconditionally across both Hugging Face (`hf-transfer`) and direct HTTPS downloads before any model file is accepted. |
| **Reflected HTML / XSS** | All user-controlled parameters in `/auth/device` and error pages are escaped via `html.escape(..., quote=True)` with restrictive `Content-Security-Policy`. |
| **Runaway Compute Unit Spend** | Pre-wake live `colab --auth oauth2 usage` check enforces `MIN_CU_BALANCE=10` (`REQUIRE_CU_BALANCE=true`), 10-minute keep-warm auto-stop, and server-side `colab --auth oauth2 sessions` readback after every `stop`. |

---

## 3. Secret Hygiene Rules

1. **Files only, never inline strings**: `.env`, `secrets/`, and `data/` are git-ignored. Secrets use `*_FILE` pointers with `0600` permissions.
2. **Automated Scanning**: Run `bash scripts/check-secrets.sh` locally before every commit; GitHub Actions runs Gitleaks across commit history on every push and PR.
