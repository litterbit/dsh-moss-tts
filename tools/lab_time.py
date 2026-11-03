"""对一份权重测「每帧耗时」（LLM 段），多次运行取中位数；同时报每帧字节数与精度。

用法：
    python tools/lab_time.py --weights weights/llm_q8x4.moss --frames 60 --repeats 3
    python tools/lab_time.py --weights weights/llm_q4x4.moss --frames 60 --threads 2
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "csrc" / "build" / "moss_llm.exe"
DEFAULT_PROMPT = ROOT / "runs" / "lab" / "zh_mid" / "prompt.txt"


def run_once(weights: Path, prompt: Path, frames: int, threads: int) -> dict:
    proc = subprocess.run([str(EXE), "--weights", str(weights), "--prompt", str(prompt),
                           "--frames", str(frames), "--threads", str(threads)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = proc.stdout or ""
    if proc.returncode != 0:
        raise SystemExit(f"失败: {proc.stderr[:400]}")
    vals = {}
    for key in ("prefill_ms", "gen_ms", "global_ms_per_frame", "local_ms_per_frame", "total_ms_per_frame"):
        m = re.search(rf'"{key}":\s*([0-9.]+)', out)
        if m:
            vals[key] = float(m.group(1))
    return vals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    weights = Path(args.weights)
    prompt = Path(args.prompt)
    runs = [run_once(weights, prompt, args.frames, args.threads) for _ in range(args.repeats)]
    # 第一次含冷启动（页缓存未命中），取后几次
    warm = runs[1:] if len(runs) > 1 else runs
    result = {
        "weights": str(weights),
        "size_mb": round(weights.stat().st_size / 1e6, 1),
        "threads": args.threads,
        "frames": args.frames,
        "per_frame_ms": round(statistics.median(r["total_ms_per_frame"] for r in warm), 2),
        "global_ms": round(statistics.median(r["global_ms_per_frame"] for r in warm), 2),
        "local_ms": round(statistics.median(r["local_ms_per_frame"] for r in warm), 2),
        "prefill_ms": round(statistics.median(r["prefill_ms"] for r in warm), 1),
        "runs": runs,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
