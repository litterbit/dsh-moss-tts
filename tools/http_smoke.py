#!/usr/bin/env python
"""本地 HTTP 服务的冒烟测试（标准库，无第三方依赖）。

    python tools/http_smoke.py --base http://127.0.0.1:8759

覆盖：
  1. /health /info 可用
  2. 同一请求两次的 PCM 逐字节一致（常驻进程内不能有跨请求状态泄漏）
  3. 取消：流式进行中 POST /jobs/{id}/cancel，连接应及时结束、任务状态为 cancelled
  4. 安全：错误 Host / 跨源 Origin / 目录穿越 / 超大 JSON 都被拒绝
  5. WAV 接口返回可用的 RIFF 文件

退出码 0 = 全过；1 = 有失败项。
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
import time
import urllib.parse

FAILS: list[str] = []

try:  # Windows 控制台默认 cp936，直接打印中文会炸；统一按 UTF-8 输出
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


class Client:
    def __init__(self, base: str):
        parsed = urllib.parse.urlparse(base)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.base = base

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None,
                timeout: float = 60.0) -> tuple[int, dict, bytes]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        code = resp.status
        head = {k.lower(): v for k, v in resp.getheaders()}
        conn.close()
        return code, head, data

    def stream(self, payload: dict, on_headers=None, timeout: float = 300.0):
        """返回 (job_id, pcm_bytes, eof_seconds)。"""
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        body = json.dumps(payload).encode("utf-8")
        conn.request("POST", "/api/v1/tts/stream", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        job_id = resp.getheader("X-Moss-Request-Id")
        if on_headers:
            on_headers(job_id)
        started = time.perf_counter()
        chunks: list[bytes] = []
        while True:
            block = resp.read(65536)
            if not block:
                break
            chunks.append(block)
        cost = time.perf_counter() - started
        conn.close()
        return job_id, b"".join(chunks), cost


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8759")
    parser.add_argument("--voice", default="Junhao")
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()
    client = Client(args.base)

    # 1. 基础接口
    code, _, body = client.request("GET", "/api/v1/health")
    check("health 200", code == 200, body.decode("utf-8", "replace")[:80])
    code, _, body = client.request("GET", "/api/v1/info")
    info = json.loads(body or b"{}")
    check("info 有 engine 字段", code == 200 and "engine" in info, str(info.get("engine")))
    code, _, body = client.request("GET", "/api/v1/voices")
    voices = json.loads(body or b"{}")
    check("voices 非空", code == 200 and len(voices.get("builtin", [])) > 0,
          f"{len(voices.get('builtin', []))} 内置 / {len(voices.get('user', []))} 用户")

    payload = {"text": "smoke test sentence.", "voice": args.voice, "seed": 7,
               "greedy": True, "max_frames": args.frames}

    # 2. 确定性
    _, first, _ = client.stream(payload)
    _, second, _ = client.stream(payload)
    check("两次请求 PCM 一致", first == second and len(first) > 0, f"{len(first)} bytes")

    # 3. 取消
    cancelled = {"job": None}
    done = threading.Event()
    timing = {}

    def stream_then_wait():
        def on_headers(job_id):
            cancelled["job"] = job_id
        started = time.perf_counter()
        job_id, pcm, _ = client.stream({"text": "a long cancellation test sentence for smoke.",
                                        "voice": args.voice, "seed": 0, "greedy": True,
                                        "max_frames": 375}, on_headers=on_headers)
        timing["eof"] = time.perf_counter() - started
        timing["pcm"] = len(pcm)
        done.set()

    thread = threading.Thread(target=stream_then_wait, daemon=True)
    thread.start()
    deadline = time.time() + 3.0
    while cancelled["job"] is None and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.6)
    if cancelled["job"] is None:
        check("取消：拿到 job id", False, "没收到 X-Moss-Request-Id")
    else:
        t0 = time.perf_counter()
        code, _, _ = client.request("POST", f"/api/v1/jobs/{cancelled['job']}/cancel", body=b"")
        check("取消接口 200", code == 200, f"HTTP {code}")
        done.wait(timeout=5.0)
        eof_ms = (time.perf_counter() - t0) * 1000.0
        state = {}
        for _ in range(40):
            code, _, body = client.request("GET", f"/api/v1/jobs/{cancelled['job']}")
            state = json.loads(body or b"{}")
            if state.get("state") in ("cancelled", "completed", "failed"):
                break
            time.sleep(0.05)
        check("取消后状态为 cancelled", state.get("state") == "cancelled", str(state.get("state")))
        check("取消后连接 1 秒内结束", eof_ms < 1000.0, f"{eof_ms:.0f} ms")
        check("取消后没有吐完整段音频", timing.get("pcm", 0) < 375 * 3840 * 4,
              f"{timing.get('pcm', 0)} bytes")

    # 4. 安全
    code, _, _ = client.request("GET", "/api/v1/health", headers={"Host": "evil.example"})
    check("错误 Host 被拒", code == 403, f"HTTP {code}")

    code, _, _ = client.request("POST", "/api/v1/tts/stream",
                                body=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json",
                                         "Origin": "https://evil.example"})
    check("跨源 Origin 被拒", code == 403, f"HTTP {code}")

    code, _, _ = client.request("GET", "/../AGENTS.md")
    check("目录穿越被拒", code in (400, 403, 404), f"HTTP {code}")

    big = json.dumps({"text": "x" * 300000}).encode()
    code, _, _ = client.request("POST", "/api/v1/tts/stream", body=big,
                                headers={"Content-Type": "application/json"})
    check("超大 JSON 被拒", code == 413, f"HTTP {code}")

    bad = b"{oops"
    code, _, _ = client.request("POST", "/api/v1/tts/stream", body=bad,
                                headers={"Content-Type": "application/json"})
    check("坏 JSON 被拒", code == 400, f"HTTP {code}")

    code, _, _ = client.request("GET", "/api/v1/tts/stream")
    check("错误方法被拒", code in (405, 404), f"HTTP {code}")

    # 5. WAV
    code, head, body = client.request("POST", "/api/v1/tts/wav",
                                      body=json.dumps(payload).encode(),
                                      headers={"Content-Type": "application/json"})
    check("wav 200", code == 200, f"HTTP {code} {len(body)} bytes")
    check("wav 头正确", body[:4] == b"RIFF" and body[8:12] == b"WAVE" and len(body) > 44)

    print()
    if FAILS:
        print(f"{len(FAILS)} 项失败：" + "、".join(FAILS))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
