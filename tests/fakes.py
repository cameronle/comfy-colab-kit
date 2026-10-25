import asyncio
import json
import re
from pathlib import Path

FAKE_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"


class FakeColab:
    """In-memory stand-in for the colab CLI. Counts calls; never touches Google."""

    def __init__(
        self,
        lines=None,
        new_delay: float = 0.02,
        exec_gate: asyncio.Event | None = None,
        balance: float | None = 100.0,
        keep_on_stop: bool = False,
    ):
        self.calls: list[str] = []
        self.live: set[str] = set()
        self.lines = (
            lines
            if lines is not None
            else [
                "KIT_STAGE bootstrap start",
                "KIT_STAGE bootstrap done 1.0",
                "KIT_STAGE download start",
                'KIT_PROGRESS download {"done_gb": 1, "total_gb": 2}',
                "KIT_STAGE comfy_boot start",
                "KIT_STAGE smoke start",
                'KIT_READY {"total_s": 5}',
            ]
        )
        self.new_delay = new_delay
        self.exec_gate = exec_gate
        self.exec_code = ""
        self.balance = balance
        self.keep_on_stop = keep_on_stop
        self.remote_files: dict[str, bytes] = {}

    async def sessions(self):
        self.calls.append("sessions")
        return sorted(self.live)

    async def new(self, name, accelerator):
        self.calls.append(f"new:{name}:{accelerator}")
        await asyncio.sleep(self.new_delay)
        self.live.add(name)

    async def upload(self, name, src, dst):
        self.calls.append(f"upload:{dst}")
        p = Path(src)
        if p.is_file():
            self.remote_files[dst] = p.read_bytes()

    async def download(self, name, remote_path, local_path):
        self.calls.append(f"download:{remote_path}")
        data = self.remote_files.get(remote_path, FAKE_PNG)
        Path(local_path).write_bytes(data)

    async def exec_stream(self, name, code, timeout_s):
        self.calls.append("exec")
        self.exec_code = code
        if self.exec_gate:
            await self.exec_gate.wait()
        m = re.search(r"run_task_file\((['\"])(.+?)\1\)", code)
        if m:
            remote_task = m.group(2)
            raw_task = json.loads(self.remote_files[remote_task].decode("utf-8"))
            pid = raw_task["prompt_id"]
            raw_str = json.dumps(raw_task)
            if "FORCE_CUDA_OOM" in raw_str:
                yield "KIT_JOB_ERROR " + json.dumps({"prompt_id": pid, "error": "CUDA out of memory", "oom": True})
                return
            yield "KIT_JOB_PROGRESS " + json.dumps({"prompt_id": pid, "node": "3", "step": 5, "max": 10})
            remote_img = f"/content/ComfyUI/output/{pid}.png"
            self.remote_files[remote_img] = FAKE_PNG
            yield "KIT_JOB_RESULT " + json.dumps(
                {
                    "prompt_id": pid,
                    "status": "success",
                    "duration_s": 0.4,
                    "outputs": {
                        "26": {
                            "images": [
                                {
                                    "filename": "out.png",
                                    "subfolder": "",
                                    "type": "output",
                                    "remote_path": remote_img,
                                }
                            ]
                        }
                    },
                }
            )
            return
        for line in self.lines:
            await asyncio.sleep(0)
            yield line

    async def restart_kernel(self, name):
        self.calls.append("restart_kernel")

    async def check_ready(self, name: str) -> bool:
        self.calls.append("check_ready")
        return name in self.live

    async def stop(self, name):
        self.calls.append("stop")
        if not self.keep_on_stop:
            self.live.discard(name)

    async def ccu_info(self):
        return self.balance
