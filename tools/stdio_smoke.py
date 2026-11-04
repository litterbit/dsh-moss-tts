#!/usr/bin/env python
"""--stdio 通道冒烟测试（JSON 行协议，给 DSH 插件用的那条路）。

    python tools/stdio_smoke.py --exe csrc/build/moss_tts.exe --models models

覆盖：ping / info / voices / speak（收齐 PCM 并统计）/ cancel / clone / 关闭 stdin 后进程退出。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


class Stdio:
    def __init__(self, exe: str, models: str):
        self.proc = subprocess.Popen(
            [exe, "--models", models, "--stdio", "--data", os.path.join("runs", "stdio_userdata")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self.lines: list[dict] = []
        self.cond = threading.Condition()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self.cond:
                self.lines.append(msg)
                self.cond.notify_all()

    def send(self, payload: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def wait_for(self, predicate, timeout: float = 120.0):
        deadline = time.time() + timeout
        with self.cond:
            while True:
                for msg in self.lines:
                    if predicate(msg):
                        return msg
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self.cond.wait(min(0.2, remaining))

    def wait_done(self, request_id: str, timeout: float = 120.0):
        return self.wait_for(lambda m: m.get("id") == request_id and m.get("type") in ("done", "error"), timeout)

    def close(self) -> int:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            return self.proc.wait(timeout=30)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", default="csrc/build/moss_tts.exe")
    parser.add_argument("--models", default="models")
    parser.add_argument("--wav", default="web/landing/assets/demo-clone.wav")
    args = parser.parse_args()

    client = Stdio(args.exe, args.models)
    try:
        client.send({"id": "p1", "cmd": "ping"})
        check("ping", client.wait_for(lambda m: m.get("id") == "p1") is not None)

        client.send({"id": "i1", "cmd": "info"})
        info = client.wait_for(lambda m: m.get("id") == "i1")
        check("info", info is not None and "engine" in (info or {}).get("info", {}), str((info or {}).get("info", {}).get("engine")))

        client.send({"id": "v1", "cmd": "voices"})
        voices = client.wait_for(lambda m: m.get("id") == "v1")
        check("voices", voices is not None and len(voices.get("builtin", [])) > 0,
              f"{len((voices or {}).get('builtin', []))} 内置")

        # 说话：收齐 audio 行，再等 done
        client.send({"id": "s1", "cmd": "speak", "text": "stdio channel test.", "voice": "Junhao",
                     "seed": 3, "greedy": True, "max_frames": 24})
        done = client.wait_done("s1")
        audio = [m for m in client.lines if m.get("id") == "s1" and m.get("type") == "audio"]
        pcm = b"".join(base64.b64decode(m["pcm"]) for m in audio)
        job = (done or {}).get("job", {})
        check("speak done", done is not None and done.get("type") == "done", str(job.get("state")))
        check("speak 有音频且与帧数一致", len(pcm) == job.get("frames", -1) * 3840 * 4,
              f"{len(pcm)} bytes / {job.get('frames')} frames")
        check("begin 在 audio 之前",
              next((i for i, m in enumerate(client.lines) if m.get("id") == "s1" and m.get("type") == "begin"), 10**9)
              < next((i for i, m in enumerate(client.lines) if m.get("id") == "s1" and m.get("type") == "audio"), 10**9))

        # 取消：min_frames 保证取消一定落在运行中
        client.send({"id": "s2", "cmd": "speak", "text": "another long test sentence.", "voice": "Junhao",
                     "seed": 0, "greedy": True, "min_frames": 200, "max_frames": 375})
        begin = client.wait_for(lambda m: m.get("id") == "s2" and m.get("type") == "begin")
        time.sleep(0.5)
        job_id = (begin or {}).get("job")
        client.send({"id": "c1", "cmd": "cancel", "job": job_id})
        cancelled = client.wait_for(lambda m: m.get("id") == "c1" and m.get("type") == "cancelled")
        done2 = client.wait_done("s2", timeout=20)
        check("cancel 返回 ok", bool((cancelled or {}).get("ok")), str(cancelled))
        check("取消后状态 cancelled", (done2 or {}).get("job", {}).get("state") == "cancelled",
              str((done2 or {}).get("job", {}).get("state")))

        # 克隆
        if os.path.exists(args.wav):
            with open(args.wav, "rb") as handle:
                payload = base64.b64encode(handle.read()).decode("ascii")
            client.send({"id": "k1", "cmd": "clone", "name": "stdio_test", "wav": payload})
            cloned = client.wait_for(lambda m: m.get("id") == "k1", timeout=120)
            check("clone", cloned is not None and cloned.get("type") == "cloned", str(cloned))
        else:
            check("clone", False, f"缺少测试 WAV：{args.wav}")

        code = client.close()
        check("stdin 关闭后进程退出 0", code == 0, f"exit={code}")
    finally:
        if client.proc.poll() is None:
            client.proc.kill()

    print()
    if FAILS:
        print(f"{len(FAILS)} 项失败：" + "、".join(FAILS))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
