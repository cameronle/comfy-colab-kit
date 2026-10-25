# comfy-colab-kit — headless Colab environment provisioner (runs via `colab exec`).
# Input:  KIT_PLAN=/content/kit/plan.json (built by gate/plan.py), optional HF_TOKEN in env.
# Output protocol on stdout (parsed by gate/session.py):
#   KIT_STAGE <name> start|done <secs>|fail      KIT_PROGRESS <key> <json>
#   KIT_READY <json timings>                     KIT_FAILED stage=<name> error=<repr>
#
# Compliant batch design:
#   - Zero outbound reverse-proxy tunnels or exposed network sockets
#   - Zero persistent HTTP servers inside the runtime
#   - Zero artificial anti-idle loops
#   - Mandatory SHA-256 verification across all model download paths (including Hugging Face)
import hashlib
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_PLAN_PATH = os.environ.get("KIT_PLAN", "/content/kit/plan.json")
PLAN = json.load(open(_PLAN_PATH)) if os.path.exists(_PLAN_PATH) else {}
COMFY = "/content/ComfyUI"
LOG = "/content/kit_logs"
RUNTIME_CFG = "/content/kit/runtime_config.json"
PORT = 8188
T0 = time.time()
TIMES = {}


class KitAbort(Exception):
    """A stage failed and KIT_FAILED was already printed."""


class Stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        print(f"KIT_STAGE {self.name} start", flush=True)
        return self

    def __exit__(self, et, ev, tb):
        d = time.time() - self.t
        TIMES[self.name] = round(d, 1)
        if et:
            print(f"KIT_STAGE {self.name} fail {d:.1f}", flush=True)
            print(f"KIT_FAILED stage={self.name} error={ev!r}"[:800], flush=True)
            raise KitAbort(self.name) from None
        print(f"KIT_STAGE {self.name} done {d:.1f}", flush=True)


def sh(cmd, check=True, quiet=False, env=None):
    if not quiet:
        print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(
        ["bash", "-lc", cmd],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line.rstrip("\n"), flush=True)
    rc = p.wait()
    if check and rc:
        raise RuntimeError(f"failed ({rc}): {cmd}")
    return rc


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "comfy-colab-kit"})
    if os.environ.get("HF_TOKEN") and url.startswith("https://huggingface.co/"):
        req.add_header("Authorization", "Bearer " + os.environ["HF_TOKEN"])
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def verify_sha256(filepath: str, expected_hex: str) -> None:
    """Verify file SHA-256 in pure Python; remove corrupted file on mismatch."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8 << 20)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest().lower()
    if actual != expected_hex.strip().lower():
        try:
            os.remove(filepath)
        except OSError:
            pass
        raise RuntimeError(f"sha256 mismatch for {filepath}: expected {expected_hex}, got {actual}")


HF_RE = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")

HF_DL_SCRIPT = r"""
import os, sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2], revision=sys.argv[3], local_dir=sys.argv[4],
                      token=os.environ.get("HF_TOKEN") or None), flush=True)
"""
HF_TMP = "/content/hf"
STALL_S = int(os.environ.get("KIT_STALL_S", "30"))
RETRYABLE = re.compile(r"429|Too Many|\b5\d\d\b|timed out|Timeout|Connection|reset by peer")


def du(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def comfy_busy():
    return False


class Downloads:
    def __init__(self, models, concurrency, busy=comfy_busy, stall_s=STALL_S):
        self.models = models
        self.size = {m["id"]: int((m.get("size_gb") or 0) * 1e9) for m in models}
        self.total = sum(self.size.values())
        self.core_ids = [m["id"] for m in models if m.get("core")]
        self.core_total = sum(self.size[i] for i in self.core_ids)
        self.done_bytes, self.core_done_bytes = 0, 0
        self.files_done = 0
        self.errors = []
        self.restarts = 0
        self.lock = threading.Lock()
        self.live = {}
        self.procs = {}
        self.paused = False
        self.busy, self.stall_s = busy, stall_s
        self.pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
        self.bg_pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
        self.core_done = threading.Event()
        self.futures = {}
        self.t0 = time.time()

    def start(self):
        order = sorted(self.models, key=lambda m: (not m.get("core"), -(m.get("size_gb") or 0)))
        seen_sha = {}
        for m in order:
            sha = m.get("sha256")
            if sha and sha in seen_sha:
                self.futures[m["id"]] = self.bg_pool.submit(self._link_after, seen_sha[sha], m)
                continue
            if sha:
                seen_sha[sha] = m["id"]
            if m.get("core"):
                self.futures[m["id"]] = self.pool.submit(self._one, m)
            else:
                self.futures[m["id"]] = self.bg_pool.submit(self._after_core, m)
        core = [self.futures[i] for i in self.core_ids if i in self.futures]
        threading.Thread(target=lambda: ([f.exception() for f in core], self.core_done.set()), daemon=True).start()
        threading.Thread(target=self._report, daemon=True).start()
        threading.Thread(target=self._pauser, daemon=True).start()

    def all_done(self):
        return self.files_done >= len(self.models) or bool(self.errors)

    def _after_core(self, m):
        self.core_done.wait()
        while self.paused:
            time.sleep(1)
        return self._one(m)

    def _link_after(self, first_id, m):
        self.futures[first_id].result()
        src = next(x for x in self.models if x["id"] == first_id)
        dst = f"{COMFY}/models/{m['dest']}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            os.link(f"{COMFY}/models/{src['dest']}", dst)
        self._finish(m)

    def _pauser(self):
        while not self.all_done():
            time.sleep(2)
            want = self.busy()
            if want == self.paused:
                continue
            self.paused = want
            for p, pausable in list(self.procs.values()):
                if pausable and p.poll() is None:
                    try:
                        p.send_signal(signal.SIGSTOP if want else signal.SIGCONT)
                    except OSError:
                        pass
            print(f"  background downloads {'paused: prompt running' if want else 'resumed'}", flush=True)

    def _hf(self, m, repo, rev, fname, dst):
        tmp = f"{HF_TMP}/{m['id']}"
        pausable = not m.get("core")
        attempt = 0
        while True:
            p = subprocess.Popen(
                [sys.executable, "-c", HF_DL_SCRIPT, repo, fname, rev, tmp],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            with self.lock:
                self.live[m["id"]] = tmp
                self.procs[m["id"]] = (p, pausable)
            if pausable and self.paused:
                p.send_signal(signal.SIGSTOP)
            out = []
            reader = threading.Thread(target=lambda: out.extend(p.stdout), daemon=True)
            reader.start()
            last, last_t, stalled = -1, time.time(), False
            while p.poll() is None:
                time.sleep(3)
                if pausable and self.paused:
                    last_t = time.time()
                    continue
                size = du(tmp)
                target = int(m.get("size_gb", 0) * 1e9)
                is_verifying = target > 0 and size >= int(target * 0.98)
                max_stall = (self.stall_s * 3) if is_verifying else self.stall_s
                if size != last:
                    last, last_t = size, time.time()
                elif time.time() - last_t > max_stall:
                    print(f"  stall: no bytes for {max_stall}s on {m['id']}; restarting that file", flush=True)
                    p.kill()
                    stalled = True
                    break
            p.wait()
            reader.join(5)
            with self.lock:
                self.procs.pop(m["id"], None)
            text = "".join(out).strip()
            if p.returncode == 0 and not stalled and text:
                path = text.splitlines()[-1].strip()
                break
            attempt += 1
            if stalled:
                self.restarts += 1
            if attempt >= 8 or (not stalled and not RETRYABLE.search(text)):
                raise RuntimeError(text[-300:] or f"exit {p.returncode}")
            wait = 2 if stalled else min(120, 10 * 2 ** (attempt - 1))
            if not stalled:
                print(f"  retry {m['id']} in {wait}s ({text[-120:]})", flush=True)
            time.sleep(wait)
        shutil.move(path, dst)
        with self.lock:
            self.live[m["id"]] = dst
        shutil.rmtree(tmp, ignore_errors=True)

    def _one(self, m):
        dst = f"{COMFY}/models/{m['dest']}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            if m.get("sha256"):
                verify_sha256(dst, m["sha256"])
            return self._finish(m)
        hm = HF_RE.match(m["url"])
        try:
            if hm:
                self._hf(m, hm.group(1), hm.group(2), hm.group(3), dst)
            else:
                rc = (
                    sh(
                        f"aria2c -q -x16 -s16 -k1M -d '{os.path.dirname(dst)}' -o '{os.path.basename(dst)}' '{m['url']}'",
                        check=False,
                        quiet=True,
                    )
                    if shutil.which("aria2c")
                    else 1
                )
                if rc:
                    with urllib.request.urlopen(m["url"], timeout=60) as r, open(dst + ".part", "wb") as f:
                        shutil.copyfileobj(r, f, 8 << 20)
                    os.replace(dst + ".part", dst)
            # Mandatory SHA-256 verification across ALL download branches (HF and HTTP)
            if m.get("sha256"):
                verify_sha256(dst, m["sha256"])
        except Exception as e:
            with self.lock:
                self.errors.append(f"{m['id']}: {e}")
            print(f"download failed {m['id']}: {e}", flush=True)
            raise
        self._finish(m)

    def _finish(self, m):
        with self.lock:
            self.live.pop(m["id"], None)
            self.files_done += 1
            self.done_bytes += self.size[m["id"]]
            if m.get("core"):
                self.core_done_bytes += self.size[m["id"]]
        print(f"  ok {m['dest']}", flush=True)

    def snapshot(self):
        with self.lock:
            live = dict(self.live)
            done, core_done = self.done_bytes, self.core_done_bytes
        inflight = {i: min(du(t), self.size[i] or 1 << 62) for i, t in live.items()}
        cur_b = done + sum(inflight.values())
        cur_cb = core_done + sum(v for i, v in inflight.items() if i in self.core_ids)
        with self.lock:
            if not hasattr(self, "_max_b"):
                self._max_b, self._max_cb = 0, 0
            self._max_b = max(self._max_b, cur_b)
            self._max_cb = max(self._max_cb, cur_cb)
            return (self._max_b, self._max_cb)

    def progress(self, rate, eta=None):
        b, cb = self.snapshot()
        left = max(0, self.total - b)
        if eta is None and rate > 1e5 and not self.paused:
            eta = int(left / rate)
        return {
            "bytes_done": b,
            "bytes_total": self.total,
            "core_bytes_done": cb,
            "core_bytes_total": self.core_total,
            "bytes_per_s": int(rate),
            "eta_s": eta,
            "files_done": self.files_done,
            "files_total": len(self.models),
            "core_done": self.core_done.is_set(),
            "paused": self.paused,
            "restarts": self.restarts,
            "done_gb": round(b / 1e9, 2),
            "total_gb": round(self.total / 1e9, 2),
        }

    def _report(self):
        last_b, last_t, rate = self.snapshot()[0], time.time(), 0.0
        while not self.all_done():
            time.sleep(5)
            b, now = self.snapshot()[0], time.time()
            inst = max(0.0, (b - last_b) / max(1e-6, now - last_t))
            rate = inst if rate == 0 else 0.6 * rate + 0.4 * inst
            last_b, last_t = b, now
            avg_rate = b / max(1.0, now - self.t0)
            effective_eta_rate = max(rate, 0.4 * avg_rate)
            left = max(0, self.total - b)
            eta = int(left / effective_eta_rate) if effective_eta_rate > 1e5 and not self.paused else None
            print("KIT_PROGRESS download " + json.dumps(self.progress(rate, eta=eta)), flush=True)
        print("KIT_PROGRESS download " + json.dumps(self.progress(0, eta=0)), flush=True)

    def wait(self, ids):
        for i in ids:
            self.futures[i].result()
        if set(self.core_ids).issubset(set(ids)):
            self.core_done.wait()


# ─────────────────────────── workflows ───────────────────────────
def png_workflow(data):
    b, i = data[8:], 0
    while i < len(b):
        (n,) = struct.unpack(">I", b[i : i + 4])
        t, d = b[i + 4 : i + 8], b[i + 8 : i + 8 + n]
        i += 12 + n
        if t == b"tEXt" and d.startswith(b"workflow\0"):
            return json.loads(d.split(b"\0", 1)[1])
    raise RuntimeError("PNG has no embedded workflow")


def remap_models(obj, aliases):
    """Rewrite loader filenames anywhere in a UI graph or API prompt."""
    if not aliases:
        return 0
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v in aliases:
                obj[k] = aliases[v]
                n += 1
            else:
                n += remap_models(v, aliases)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and v in aliases:
                obj[i] = aliases[v]
                n += 1
            else:
                n += remap_models(v, aliases)
    return n


def decorate_app_nodes(g, wf):
    nodes = {n["id"]: n for n in g.get("nodes", [])}
    for item in wf.get("app_inputs", []):
        nid, wname = item[0], item[1]
        node = nodes.get(nid)
        if not node:
            continue
        ntype = str(node.get("type", ""))
        curr_title = str(node.get("title") or "")

        label = None
        new_title = None
        if wname == "image":
            label = "参考底图 (Input Image)"
            new_title = "参考底图 (Input Image)"
        elif wname == "denoise":
            label = "重绘幅度 (Denoise Strength 0.0-1.0)"
            new_title = "重绘幅度 (Denoise)"
        elif wname == "choice":
            label = "画质与速度预设 (Preset)"
            new_title = "画质与速度预设 (Preset)"
        elif wname in ("seed", "noise_seed"):
            label = "随机种子 (Seed)"
            new_title = "随机种子 (Seed)"
        elif wname == "width":
            label = "图像宽度 (Width)"
            new_title = "图像尺寸 (Image Size)"
        elif wname == "height":
            label = "图像高度 (Height)"
            new_title = "图像尺寸 (Image Size)"
        elif wname == "batch_size":
            label = "生成张数 (Batch Size)"
            new_title = "图像尺寸 (Image Size)"
        elif wname == "switch":
            label = "提示词增强开关 (Prompt Enhancer)"
            new_title = "提示词增强 (Prompt Enhancer)"
        elif wname == "value":
            if ntype == "PrimitiveInt" or "seed" in curr_title.lower():
                label = "随机种子 (Seed)"
                new_title = "随机种子 (Seed)"
            elif "width" in curr_title.lower():
                label = "宽度 (Width)"
                new_title = "宽度 (Width)"
            elif "height" in curr_title.lower():
                label = "高度 (Height)"
                new_title = "高度 (Height)"
            else:
                label = "正向提示词 (Positive Prompt)"
                new_title = "正向提示词 (Positive Prompt)"
        elif wname == "text":
            if "neg" in curr_title.lower() or nid in (8, 4):
                label = "负面提示词 (Negative Prompt)"
                new_title = "负面提示词 (Negative Prompt)"
            else:
                label = "正向提示词 (Positive Prompt)"
                new_title = "正向提示词 (Positive Prompt)"

        generic_titles = {
            "",
            "none",
            "cliptextencode",
            "primitivestringmultiline",
            "primitiveint",
            "emptyflux2latentimage",
            "randomnoise",
            "basicscheduler",
            "loadimage",
            "customcombo",
        }
        if new_title and (not curr_title or curr_title.lower() in generic_titles):
            node["title"] = new_title

        if label:
            inputs = node.setdefault("inputs", [])
            found = False
            for slot in inputs:
                if slot.get("name") == wname or slot.get("widget", {}).get("name") == wname:
                    slot["label"] = label
                    found = True
            if not found:
                inputs.append({"name": wname, "type": "STRING", "label": label})

            widgets = node.setdefault("widgets", [])
            w_found = False
            for w in widgets:
                if isinstance(w, dict) and w.get("name") == wname:
                    w["label"] = label
                    w_found = True
            if not w_found:
                widgets.append({"name": wname, "label": label})


def prepare_workflow(wf, data, default_view):
    g = png_workflow(data) if wf["format"] == "png" else json.loads(data)
    nodes = {n["id"]: n for n in g.get("nodes", [])}
    for o in wf.get("overrides", []):
        n = nodes.get(o["node"])
        if not n or len(n.get("widgets_values") or []) <= o["widget_index"]:
            raise RuntimeError(f"{wf['id']}: override target {o} not found (upstream workflow changed?)")
        n["widgets_values"][o["widget_index"]] = o["value"]
    remap_models(g, PLAN.get("model_aliases") or {})
    decorate_app_nodes(g, wf)
    extra = g.setdefault("extra", {})
    if wf.get("app_outputs"):
        extra["linearData"] = {
            "inputs": [list(x) for x in wf.get("app_inputs", [])],
            "outputs": list(wf["app_outputs"]),
        }
        extra["linearMode"] = default_view == "app"
    return g


def install_workflows(update_default=True):
    d = f"{COMFY}/user/default/workflows"
    os.makedirs(d, exist_ok=True)
    default_file = None
    for wf in PLAN["workflows"]:
        data = wf["inline"].encode() if wf.get("inline") else fetch(wf["url"])
        g = prepare_workflow(wf, data, PLAN.get("default_view", "app"))
        for u in wf.get("loras") or []:
            dst = f"{COMFY}/models/loras/{u.rsplit('/', 1)[1]}"
            if not os.path.exists(dst):
                with open(dst + ".part", "wb") as f:
                    f.write(fetch(u, timeout=600))
                os.replace(dst + ".part", dst)
        name = f"{wf['id']}.json"
        json.dump(g, open(f"{d}/{name}", "w"), ensure_ascii=False)
        if wf.get("default"):
            default_file = name
        print(f"  workflow {name} ({wf['title']})", flush=True)
    if update_default:
        workflows_summary = [
            {"id": w["id"], "title": w.get("title", w["id"]), "file": f"{w['id']}.json"}
            for w in PLAN.get("workflows", [])
        ]
        json.dump(
            {
                "default_workflow": default_file,
                "default_view": PLAN.get("default_view", "app"),
                "workflows": workflows_summary,
            },
            open(f"{COMFY}/user/default/kit.json", "w"),
        )
        if default_file and os.path.exists(f"{d}/{default_file}"):
            shutil.copy(f"{d}/{default_file}", f"{COMFY}/default_workflow.json")


def apply_settings():
    p = f"{COMFY}/user/default/comfy.settings.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cur = json.load(open(p)) if os.path.exists(p) else {}
    cur.update(PLAN["comfyui"].get("settings") or {})
    json.dump(cur, open(p, "w"), indent=1)


def install_custom_nodes():
    base = f"{COMFY}/custom_nodes"
    os.makedirs(base, exist_ok=True)
    for n in PLAN.get("custom_nodes", []):
        dst = f"{base}/{n['id']}"
        if os.path.isdir(dst):
            continue
        if n["url"].startswith("https://huggingface.co/"):
            repo = n["url"].split("huggingface.co/", 1)[1].strip("/")
            from huggingface_hub import snapshot_download

            tmp = snapshot_download(
                repo,
                allow_patterns=[f"{n['subdir']}/*"] if n.get("subdir") else None,
                revision=n.get("ref") or None,
                local_dir=f"/content/hf/node_{n['id']}",
            )
            shutil.copytree(os.path.join(tmp, n.get("subdir") or ""), dst)
        else:
            sh(f"git clone -q {n['url']} {dst}" + (f" && cd {dst} && git checkout -q {n['ref']}" if n.get("ref") else ""))
            if n.get("subdir"):
                inner = f"{dst}.tmp"
                os.rename(dst, inner)
                shutil.copytree(f"{inner}/{n['subdir']}", dst)
                shutil.rmtree(inner)
        if os.path.exists(f"{dst}/requirements.txt"):
            sh(
                f"uv pip install --system -q -r {dst}/requirements.txt || pip install -q -r {dst}/requirements.txt",
                check=False,
            )


def comfy_cmd():
    args = ["python", "main.py", "--listen", "127.0.0.1", "--port", str(PORT), "--disable-auto-launch"]
    emp = PLAN["comfyui"].get("extra_model_paths")
    if emp:
        path = f"{COMFY}/extra_model_paths.kit.yaml"
        json.dump(emp, open(path, "w"))
        args += ["--extra-model-paths-config", path]
    args += list(PLAN["comfyui"].get("args") or [])
    return args


def write_runtime_config():
    os.makedirs(os.path.dirname(RUNTIME_CFG), exist_ok=True)
    emp = PLAN["comfyui"].get("extra_model_paths")
    emp_path = None
    if emp:
        emp_path = f"{COMFY}/extra_model_paths.kit.yaml"
        json.dump(emp, open(emp_path, "w"))
    cfg_payload = {
        "profile": PLAN.get("profile"),
        "comfy_dir": COMFY,
        "env": {k: str(v) for k, v in (PLAN["comfyui"].get("env") or {}).items()},
        "args": list(PLAN["comfyui"].get("args") or []),
        "extra_model_paths_file": emp_path,
        "model_aliases": PLAN.get("model_aliases") or {},
    }
    with open(RUNTIME_CFG, "w") as f:
        json.dump(cfg_payload, f, indent=2)


def smoke_prompt(d):
    if isinstance(d, dict) and isinstance(d.get("prompt"), dict):
        return d["prompt"]
    return d


def smoke():
    import importlib.util

    api = smoke_prompt(json.loads(fetch(PLAN["smoke"]["api_url"])))
    remap_models(api, PLAN.get("model_aliases") or {})
    runner_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "headless_runner.py")
    spec = importlib.util.spec_from_file_location("kit_headless_runner", runner_file)
    assert spec and spec.loader
    headless_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(headless_runner)
    res = headless_runner.execute_prompt(api, prompt_id="smoke", runtime_cfg_path=RUNTIME_CFG)
    if not any(v.get("images") for v in (res.get("outputs") or {}).values()):
        raise RuntimeError("smoke execution finished without output images")


def fetch_only():
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    if PLAN.get("xet_high_performance", True):
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    ids = [w["id"] for w in PLAN["workflows"]]
    try:
        install_custom_nodes()
        install_workflows(update_default=False)
        dl = Downloads(PLAN["models"], PLAN.get("download_concurrency", 8))
        dl.start()
        dl.wait([m["id"] for m in PLAN["models"]])
    except Exception as e:  # noqa: BLE001
        print(f"KIT_FETCH_FAILED {json.dumps({'workflows': ids, 'error': repr(e)[:300]})}", flush=True)
        return
    print(f"KIT_FETCHED {json.dumps({'workflows': ids})}", flush=True)


def main():
    os.makedirs(LOG, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(RUNTIME_CFG), "tasks"), exist_ok=True)
    os.makedirs(f"{COMFY}/input", exist_ok=True)
    os.makedirs(f"{COMFY}/output", exist_ok=True)
    if PLAN.get("fetch_only"):
        return fetch_only()
    cfg = PLAN["comfyui"]
    if os.path.isfile(RUNTIME_CFG) and os.path.isdir(f"{COMFY}/comfy"):
        print("Headless ComfyUI environment already prepared on this VM; reusing.", flush=True)
        print("KIT_READY " + json.dumps({"reused": True}), flush=True)
        return
    with Stage("bootstrap"):
        sh("pip install -q -U uv 'huggingface_hub>=0.36' 'hf-transfer>=0.1.8' 'hf-xet>=1.5.2'", quiet=True)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        if PLAN.get("xet_high_performance", True):
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        if not os.path.isdir(f"{COMFY}/.git"):
            sh(f"rm -rf {COMFY} && git clone -q {cfg['repo']} {COMFY} && cd {COMFY} && git checkout -q {cfg['commit']}")
        sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)
    dl = Downloads(PLAN["models"], PLAN.get("download_concurrency", 8))
    dl.start()
    try:
        with Stage("deps"):
            frontend = cfg.get("frontend_package")
            sh(
                f"cd {COMFY} && grep -viE '^(torch|torchvision|torchaudio)([<>=~ ]|$)' requirements.txt > /tmp/req.txt && "
                + (
                    f"(uv pip install --system -q -r /tmp/req.txt {repr(frontend) if frontend else ''} || "
                    if PLAN.get("pip_installer", "uv") == "uv"
                    else "("
                )
                + f"pip install -q -r /tmp/req.txt {repr(frontend) if frontend else ''})"
            )
            install_custom_nodes()
        with Stage("install_workflows"):
            install_workflows()
            apply_settings()
        with Stage("download"):
            # Wait for all planned models before exiting setup so background threads complete cleanly
            dl.wait([m["id"] for m in PLAN["models"]])
        with Stage("comfy_boot"):
            write_runtime_config()
            # Drop HF_TOKEN from kernel environment after downloads finish so untrusted nodes never see it
            os.environ.pop("HF_TOKEN", None)
        if PLAN.get("smoke"):
            with Stage("smoke"):
                smoke()
        print(
            "KIT_READY "
            + json.dumps(
                {
                    "total_s": round(time.time() - T0),
                    "stages": TIMES,
                    "background_downloads": 0,
                }
            ),
            flush=True,
        )
    finally:
        dl.pool.shutdown(wait=False, cancel_futures=False)
        dl.bg_pool.shutdown(wait=False, cancel_futures=False)


if __name__ == "__main__":
    try:
        main()
    except KitAbort:
        pass
