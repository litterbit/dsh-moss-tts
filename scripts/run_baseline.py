"""baseline：官方 ONNX 运行时，CPU 4 线程，固定 seed，跑两次验证确定性。

    python scripts/run_baseline.py

产物：runs/baseline/zh_seed0.wav、zh_seed0.frames.npy、zh_seed0.metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_moss_tts.onnx_backend import OnnxBackend  # noqa: E402
from dsh_moss_tts.paths import run_dir  # noqa: E402

TEXT = "你好，我是语音助手，这是一段用于对齐基准的测试音频。"
THREADS = 4


def main() -> int:
    out = run_dir("baseline")
    backend = OnnxBackend(threads=THREADS, provider="cpu", output_dir=out)

    first = backend.synthesize(TEXT, tag="zh_seed0")
    second = backend.synthesize(TEXT, tag="zh_seed0_run2")
    deterministic = first.frames.shape == second.frames.shape and bool(
        (first.frames == second.frames).all()
    )

    report = {
        "provider": backend.provider,
        "threads": backend.threads,
        "sample_mode": backend.sample_mode,
        "load_seconds": round(backend.load_seconds, 2),
        "deterministic": deterministic,
        "run1": first.to_dict(),
        "run2": second.to_dict(),
    }
    (out / "zh_seed0.metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
