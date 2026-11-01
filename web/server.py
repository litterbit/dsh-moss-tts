#!/usr/bin/env python
"""moss-tts 试听网页（只用标准库，不碰推理代码）。

    python web/server.py [--port 8759]

页面：http://127.0.0.1:8759/
接口：
    GET  /api/voices                 列出内置音色 + 克隆音色
    POST /api/synth  {text, voice, kind, seed, greedy, frames}
    GET  /audio/<file>.wav           回放/下载

底层直接调发布包里的 dist/moss-tts-cpu-win64/moss_tts.exe，一次请求一个进程。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "moss-tts-cpu-win64"
EXE = DIST / "moss_tts.exe"
MODELS = DIST / "models"
BUILTIN_DIR = MODELS / "voices"
CLONE_DIR = ROOT / "web" / "voices"
OUT_DIR = ROOT / "web" / "out"
WEB_DIR = ROOT / "web"

SYNTH_LOCK = threading.Lock()
KEEP_WAVS = 24


def list_builtin() -> list[str]:
    return sorted(p.stem for p in BUILTIN_DIR.glob("*.mossclone"))


def list_clones() -> list[dict]:
    items = []
    for path in sorted(CLONE_DIR.glob("*.mossclone")):
        frames = 0
        try:
            data = path.read_bytes()
            if data[:4] == b"MSCL":
                frames = int.from_bytes(data[12:16], "little")
        except OSError:
            pass
        items.append({"name": path.stem, "frames": frames, "seconds": round(frames * 0.08, 2)})
    return items


def prune_outputs() -> None:
    wavs = sorted(OUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in wavs[KEEP_WAVS:]:
        try:
            old.unlink()
        except OSError:
            pass


def trim_trailing_babble(path: Path, peak_drop_db: float = 28.0, floor_db: float = -50.0,
                         tail_ms: int = 150) -> tuple[float, float]:
    """模型偶尔会在句尾继续低声“自言自语”（-45 dB 上下）。按能量把尾巴切掉。

    只动发布产物，不碰推理：16-bit 立体声 48 kHz，按 80 ms 一帧算 RMS，
    以「峰值 - 28 dB」为静音门限，最后一个超过门限的帧之后再留 150 ms。
    """
    import wave

    try:
        with wave.open(str(path), "rb") as handle:
            params = handle.getparams()
            raw = handle.readframes(handle.getnframes())
    except Exception:  # noqa: BLE001
        return (0.0, 0.0)
    if params.sampwidth != 2 or params.nchannels < 1 or params.framerate <= 0:
        return (0.0, 0.0)

    import array
    import math

    samples = array.array("h")
    samples.frombytes(raw)
    channels = params.nchannels
    per = int(0.08 * params.framerate)
    if per <= 0:
        return (0.0, 0.0)
    stride = per * channels
    frames = len(samples) // stride
    if frames < 3:
        return (0.0, 0.0)

    db = []
    for i in range(frames):
        seg = samples[i * stride:(i + 1) * stride]
        acc = 0
        for v in seg:
            acc += v * v
        rms = math.sqrt(acc / max(1, len(seg))) / 32768.0
        db.append(20.0 * math.log10(max(rms, 1e-9)))

    peak = max(db)
    threshold = max(floor_db, peak - peak_drop_db)
    last = -1
    for i, value in enumerate(db):
        if value >= threshold:
            last = i
    before = len(samples) / channels / params.framerate
    if last < 0:
        return (before, before)
    keep = min(len(samples), (last + 1) * stride + int(tail_ms / 1000.0 * params.framerate) * channels)
    if keep >= len(samples) or (len(samples) - keep) < params.framerate * 0.1 * channels:
        return (before, before)
    with wave.open(str(path), "wb") as handle:
        handle.setparams(params)
        handle.writeframes(samples[:keep].tobytes())
    return (before, keep / channels / params.framerate)


def parse_stats(stdout: str) -> dict:
    """发布包打印的 JSON 里路径是单反斜杠（不合法 JSON），只按字段名抽数字。"""
    stats: dict[str, float | str] = {}
    for key, value in re.findall(r'"(\w+)":\s*(-?[\d.]+)', stdout):
        stats[key] = float(value)
    for key in ("voice",):
        m = re.search(r'"voice":\s*"([^"]*)"', stdout)
        if m:
            stats[key] = m.group(1)
    return stats


def synthesize(text: str, voice: str, kind: str, seed: int, greedy: bool, frames: int,
               trim: bool = True) -> dict:
    if not EXE.exists() or not MODELS.exists():
        raise RuntimeError(f"找不到发布包：{DIST}（先跑 scripts\\pack_release.bat）")
    if not text.strip():
        raise ValueError("文本为空")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{uuid.uuid4().hex}.wav"
    cmd = [str(EXE), "--models", str(MODELS), "--text", text, "--out", str(out_path),
           "--seed", str(seed), "--frames", str(max(1, frames))]
    if kind == "clone":
        profile = CLONE_DIR / f"{voice}.mossclone"
        if not profile.exists():
            raise ValueError(f"找不到克隆音色：{voice}")
        cmd += ["--profile", str(profile)]
    else:
        cmd += ["--voice", voice]
    if greedy:
        cmd.append("--greedy")

    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if proc.returncode != 0 or not out_path.exists():
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"推理失败（exit {proc.returncode}）")

    stats = parse_stats(proc.stdout)
    if trim:
        before, after = trim_trailing_babble(out_path)
        if after > 0.0 and after < before:
            stats["trimmed_seconds"] = round(before - after, 2)
            stats["audio_seconds"] = round(after, 2)
            stats["rtf"] = round(stats["total_ms"] / max(1.0, after * 1000.0), 4)
    stats["elapsed_ms"] = round(elapsed_ms, 1)
    stats["voice"] = voice
    stats["kind"] = kind
    prune_outputs()
    return {"url": f"/audio/{out_path.name}", "file": out_path.name, "stats": stats}


class Handler(BaseHTTPRequestHandler):
    server_version = "moss-tts-web/1.0"

    def log_message(self, fmt: str, *args) -> None:  # 精简日志
        print(f"[web] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/voices":
            self._json(200, {"builtin": list_builtin(), "clones": list_clones(),
                             "ready": EXE.exists() and MODELS.exists()})
        elif path.startswith("/audio/"):
            name = path[len("/audio/"):]
            if not re.fullmatch(r"[0-9a-f]{32}\.wav", name):
                self._json(400, {"error": "bad name"})
                return
            target = OUT_DIR / name
            if not target.exists():
                self._json(404, {"error": "not found"})
                return
            self._send(200, target.read_bytes(), "audio/wav")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/synth":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            text = str(payload.get("text", ""))
            voice = str(payload.get("voice", "Junhao"))
            kind = str(payload.get("kind", "builtin"))
            seed = int(payload.get("seed", 0))
            greedy = bool(payload.get("greedy", False))
            frames = int(payload.get("frames", 375))
            trim = bool(payload.get("trim", True))
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": f"请求格式不对：{exc}"})
            return
        try:
            with SYNTH_LOCK:
                result = synthesize(text, voice, kind, seed, greedy, frames, trim)
            self._json(200, result)
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": str(exc)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8759)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[web] http://{args.host}:{args.port}/", flush=True)
    print(f"[web] exe   {EXE}", flush=True)
    print(f"[web] 内置 {len(list_builtin())} 个音色，克隆 {len(list_clones())} 个", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
