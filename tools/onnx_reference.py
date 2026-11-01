"""用 ONNX 运行时跑 greedy 基线，导出 prompt 行与 codes，供 C++ 引擎对拍。

输出：
    <out>/onnx_codes.txt   每行 16 个 code
    <out>/onnx_rows.txt    实际喂给模型的 prompt 行（用于校验与 C++ 的 prompt.txt 一致）
    <out>/onnx_meta.json   计时与配置
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "moss_tts_nano_onnx"))

from onnx_tts_runtime import OnnxTtsRuntime  # noqa: E402

MODELS = Path(r"D:\workplace\audio\github")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="Junhao")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runtime = OnnxTtsRuntime(
        model_dir=str(MODELS),
        output_dir=str(out),
        thread_count=args.threads,
        max_new_frames=args.frames,
        sample_mode="greedy",
        execution_provider="cpu",
    )

    prepared = runtime.prepare_synthesis_text(
        text=args.text, voice=args.voice, enable_wetext=False, enable_normalize_tts_text=True
    )
    prepared_text = str(prepared["text"])
    text_ids = [int(t) for t in runtime.sp_model.encode(prepared_text, out_type=int)]
    prompt_audio_codes = runtime.resolve_prompt_audio_codes(voice=args.voice, prompt_audio_path=None)
    rows = runtime.build_voice_clone_request_rows(prompt_audio_codes, text_ids)["inputIds"]

    started = time.perf_counter()
    result = runtime.synthesize(
        text=args.text,
        voice=args.voice,
        output_audio_path=out / "onnx_greedy.wav",
        sample_mode="greedy",
        do_sample=False,
        streaming=False,
        max_new_frames=args.frames,
        enable_wetext=False,
        seed=0,
    )
    elapsed = time.perf_counter() - started

    codes = result["audio_token_ids"]
    (out / "onnx_codes.txt").write_text(
        "\n".join(" ".join(str(int(v)) for v in row) for row in codes) + "\n", encoding="utf-8"
    )
    (out / "onnx_rows.txt").write_text(
        "\n".join(" ".join(str(int(v)) for v in row) for row in rows) + "\n", encoding="utf-8"
    )
    (out / "onnx_meta.json").write_text(
        json.dumps(
            {
                "text": args.text,
                "prepared_text": prepared_text,
                "text_tokens": len(text_ids),
                "rows": len(rows),
                "frames": int(codes.shape[0]),
                "synth_seconds": round(elapsed, 3),
                "sample_mode": "greedy",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out), "rows": len(rows), "frames": int(codes.shape[0]),
                      "seconds": round(elapsed, 3)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
