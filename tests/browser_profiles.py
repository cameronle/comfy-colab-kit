"""Optional real-Chromium UI acceptance: python -m tests.browser_profiles.

Uses an isolated local FakeColab server only. Never contacts Google or production.
Requires playwright and a system Chromium; no model/torch/GPU dependency.
"""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import struct
import zlib

from aiohttp import web
from playwright.async_api import async_playwright

from gate.app import BATCH, build_app
from gate.config import Settings
from gate.profiles import load_profile
from tests.fakes import FakeColab

ROOT = Path(__file__).resolve().parent.parent


def synthetic_png():
    """Decodable, visibly synthetic test pattern; not a generated model image."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + bytes([70 + 12 * (y // 16), 140, 190]) * 96 for y in range(96))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 96, 96, 8, 2, 0, 0, 0))
            + chunk(b"tEXt", b"Description\0LOCAL FAKE; NO GPU INFERENCE")
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class BrowserFakeColab(FakeColab):
    async def exec_stream(self, name, code, timeout_s):
        async for line in super().exec_stream(name, code, timeout_s):
            if line.startswith("KIT_JOB_RESULT "):
                for path in self.remote_files:
                    if path.endswith(".png"):
                        self.remote_files[path] = synthetic_png()
            yield line


async def main():
    results = []
    out = Path(os.environ.get("KIT_BROWSER_EVIDENCE", tempfile.gettempdir())) / "comfy-profile-browser"
    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=os.environ.get("CHROMIUM_BIN", "/usr/bin/chromium"),
                                           headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        try:
            for viewport in ({"width": 1366, "height": 900}, {"width": 390, "height": 844}):
                for pid, preset in (("moss-image", "Fast alt"), ("qwen-image", "Turbo"), ("sd-latest", "Quality")):
                    with tempfile.TemporaryDirectory(prefix="comfy-browser-") as state:
                        settings = Settings.from_env({"AUTH_DISABLED": "true", "STATE_DIR": state,
                                                      "DEFAULT_PROFILE": "moss-image"})
                        fake = BrowserFakeColab()
                        profiles = {key: load_profile(ROOT / "profiles/examples" / key)
                                    for key in ("moss-image", "qwen-image", "sd-latest")}
                        app = build_app(settings, fake, profiles, run_background=False)
                        # End persistent SSE handlers promptly during fixture teardown.
                        runner = web.AppRunner(app, shutdown_timeout=1)
                        await runner.setup()
                        site = web.TCPSite(runner, "127.0.0.1", 0)
                        await site.start()
                        port = site._server.sockets[0].getsockname()[1]
                        context = await browser.new_context(viewport=viewport)
                        page = await context.new_page()
                        errors = []
                        requests = []
                        page.on("pageerror", lambda err: errors.append(str(err)))
                        page.on("request", lambda r: requests.append(r.post_data_json)
                                if r.url.endswith("/_gate/txt2img") else None)
                        try:
                            await page.goto(f"http://127.0.0.1:{port}/")
                            await page.evaluate("""() => {
                                const banner = document.createElement('p');
                                banner.textContent = 'LOCAL FAKE TEST — NO GPU / 模拟流程，非真实出图';
                                banner.style.cssText = 'text-align:center;color:#b00;font-weight:bold';
                                document.body.prepend(banner);
                            }""")
                            await page.locator("#profile-select").select_option(pid)
                            options = await page.locator("#preset-select").inner_text()
                            assert preset in options, (pid, options)
                            assert ("Fast alt" in options) == (pid == "moss-image"), (pid, options)
                            assert await page.locator("#ref-file").is_visible() == (pid == "moss-image")
                            assert await page.locator("#negative-options").is_visible() == (pid != "qwen-image")
                            prompt = "  原样保留长中文提示词🌟\n" + "场景细节、主体与构图；" * 35
                            await page.locator("#prompt-input").fill(prompt)
                            await page.locator("#seed-input").fill("0")
                            await page.locator("#submit-quick-btn").click()
                            for _ in range(200):
                                jobs = list(app[BATCH]._jobs.values())
                                if jobs and jobs[0].done_event.is_set():
                                    break
                                await asyncio.sleep(0.02)
                            assert jobs and jobs[0].status == "success", [j.to_summary() for j in jobs]
                            job = jobs[0]
                            assert job.profile_id == pid
                            assert requests[0]["prompt"] == prompt
                            assert requests[0]["seed"] == 0
                            assert ("denoise" in requests[0]) is False  # no reference image selected
                            assert ("negative_prompt" in requests[0]) == (pid != "qwen-image")
                            node, field = {"moss-image": ("6", "value"), "qwen-image": ("474", "on_false"),
                                           "sd-latest": ("16", "text")}[pid]
                            assert job.prompt[node]["inputs"][field] == prompt
                            assert not errors, errors
                            assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "horizontal overflow"
                            await page.wait_for_function("""() => {
                                const img = document.querySelector('.job-thumb img');
                                return img && img.complete && img.naturalWidth === 96;
                            }""")
                            await page.wait_for_timeout(3200)  # allow the transient enqueue toast to disappear
                            await page.evaluate("window.scrollTo(0, 0)")
                            await page.screenshot(path=str(out / f"{pid}-{viewport['width']}.png"), full_page=True)
                            results.append({"profile": pid, "viewport": viewport, "status": job.status,
                                            "workflow": job.workflow_id, "prompt_unchanged": True, "seed": 0,
                                            "fake_backend": True, "page_errors": errors})
                        finally:
                            await context.close()
                            await runner.cleanup()
        finally:
            await browser.close()
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(results), "evidence": str(out), "results": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
