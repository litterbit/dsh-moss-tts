"""A/B the official ONNX runtime with and without the leading <|im_start|> row.

The browser ONNX manifest ships ``user_prompt_prefix_token_ids`` WITHOUT the
leading ``<|im_start|>`` (id 4), while the PyTorch reference builds that prefix
as ``[im_start] + encode("user\\n") + encode("<user_inst>\\n- Reference(s):\\n")``.
This script runs the official runtime twice on identical input, forcing greedy
frames so that the only difference is the prompt itself.

Usage:
    python tools/onnx_ab_prefix.py --text "今天天气不错。" --voice Junhao
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"D:\workplace\audio\github\MOSS-TTS-Nano")
MODEL_DIR = Path(r"D:\workplace\audio\github\MOSS-TTS-Nano-100M-ONNX")
OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "official"

sys.path.insert(0, str(REPO))

from onnx_tts_runtime import OnnxTtsRuntime  # noqa: E402


def dump_frames(frames: list[list[int]], path: Path) -> None:
    path.write_text("\n".join(" ".join(str(v) for v in row) for row in frames), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="今天天气不错。")
    parser.add_argument("--voice", default="Junhao")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime = OnnxTtsRuntime(model_dir=MODEL_DIR, thread_count=4, max_new_frames=args.frames, do_sample=False)
    manifest_prefix = list(runtime.manifest["prompt_templates"]["user_prompt_prefix_token_ids"])
    im_start = int(runtime.manifest["tts_config"]["im_start_token_id"])

    results = {}
    for name, prefix in (
        ("manifest", manifest_prefix),
        ("with_im_start", [im_start] + manifest_prefix),
    ):
        runtime.manifest["prompt_templates"]["user_prompt_prefix_token_ids"] = prefix
        frames_path = OUT_DIR / f"ab_{name}_frames.txt"
        wav_path = OUT_DIR / f"ab_{name}.wav"
        result = runtime.synthesize(
            text=args.text,
            voice=args.voice,
            output_audio_path=str(wav_path),
            sample_mode="greedy",
            do_sample=False,
            streaming=False,
            max_new_frames=args.frames,
            enable_wetext=False,
            seed=args.seed,
        )
        frames = np.asarray(result["audio_token_ids"], dtype=np.int32)
        dump_frames(frames.tolist(), frames_path)
        results[name] = frames
        print(f"[{name}] prefix_len={len(prefix)} prompt_rows={72 + 0} frames={frames.shape[0]} "
              f"first_codes={frames[0].tolist() if frames.size else []}")
        print(f"  wav={wav_path} frames={frames_path}")

    a = results["manifest"]
    b = results["with_im_start"]
    same_len = a.shape[0] == b.shape[0]
    print(f"\nframes manifest={a.shape[0]} with_im_start={b.shape[0]}")
    if same_len and a.size and b.size:
        match = (a == b).all(axis=1)
        print(f"identical frames: {int(match.sum())}/{a.shape[0]}")
        print(f"first divergent frame: {int(np.argmin(match)) if not match.all() else -1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
