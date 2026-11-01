"""Compare the rendered prompt-fix A/B wavs (durations, levels, onset correlation)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1] / "runs" / "promptfix"


def load(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as handle:
        frames = handle.getnframes()
        channels = handle.getnchannels()
        rate = handle.getframerate()
        data = np.frombuffer(handle.readframes(frames), dtype=np.int16).astype(np.float32) / 32768.0
    return data.reshape(-1, channels), rate


def main() -> int:
    for name in ("torch_greedy", "cpp_greedy", "cpp_greedy_nocache", "torch_sample_s0", "cpp_sample_s0"):
        path = BASE / f"{name}.wav"
        if not path.exists():
            continue
        audio, rate = load(path)
        peak = float(np.abs(audio).max())
        rms = float(np.sqrt((audio**2).mean()))
        print(f"{name:20s} {len(audio) / rate:5.2f}s peak={peak:.3f} rms={rms:.3f} frames~{len(audio) / 3840:.0f}")

    reference = BASE / "torch_greedy.wav"
    candidate = BASE / "cpp_greedy_nocache.wav"
    if reference.exists() and candidate.exists():
        a, rate = load(reference)
        b, _ = load(candidate)
        for label, start, end in (("0-80ms", 0, 3840), ("80-160ms", 3840, 7680), ("160-320ms", 7680, 15360)):
            xa = a[start:end].ravel()
            xb = b[start:end].ravel()
            if xa.size == 0 or xb.size == 0:
                continue
            corr = float(np.sum(xa * xb) / (np.sqrt(np.sum(xa * xa)) * np.sqrt(np.sum(xb * xb)) + 1e-12))
            print(f"onset corr {label:10s} r={corr:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
