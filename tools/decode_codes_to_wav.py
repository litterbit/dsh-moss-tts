"""Decode a codes text file (one frame per line, 16 ints) to a 48 kHz stereo wav.

Uses the official ONNX codec so the audio is comparable across backends.

    python tools/decode_codes_to_wav.py --codes runs/cpp/codes.txt --out runs/x.wav
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_moss_tts.paths import MODELS_DIR, TTS_REPO_DIR  # noqa: E402

ONNX_TTS_DIR = MODELS_DIR / "MOSS-TTS-Nano-100M-ONNX"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default=str(ONNX_TTS_DIR))
    args = parser.parse_args()

    frames = [
        [int(v) for v in line.split()]
        for line in Path(args.codes).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sys.path.insert(0, str(TTS_REPO_DIR))
    from onnx_tts_runtime import OnnxTtsRuntime

    runtime = OnnxTtsRuntime(model_dir=args.models, thread_count=4)
    channels, _length = runtime.decode_full_audio(frames)
    sample_rate = 48000
    length = min(len(channel) for channel in channels)
    pcm = np.stack([np.asarray(channel[:length], dtype=np.float32) for channel in channels], axis=1)
    pcm16 = np.round(np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as handle:
        handle.setnchannels(pcm16.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())
    print(f"{len(frames)} frames -> {out} ({length / sample_rate:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
