"""Compare the codec paths: PyTorch encode vs ONNX encode vs the shipping manifest codes.

If the PyTorch codec produces different codes for the same wav than the ONNX codec,
the PyTorch reference in this environment is not trustworthy, and any audio rendered
from its prompt codes will be noise.

Usage:
    python tools/check_codec_paths.py --wav assets/audio/zh_1.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torch_common import ONNX_TTS_DIR, REPO_DIR, encode_reference, load_torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default=str(REPO_DIR / "assets" / "audio" / "zh_1.wav"))
    parser.add_argument("--voice", default="Junhao")
    args = parser.parse_args()

    torch, model, _tokenizer, codec = load_torch(attn_implementation="eager")
    codes_torch, sample_rate, channels = encode_reference(model, codec, torch, args.wav)
    print(f"torch encode: {codes_torch.shape} sample_rate={sample_rate} channels={channels}")

    sys.path.insert(0, str(REPO_DIR))
    from onnx_tts_runtime import OnnxTtsRuntime

    runtime = OnnxTtsRuntime(model_dir=ONNX_TTS_DIR, thread_count=4)
    codes_onnx = np.asarray(runtime.encode_reference_audio(args.wav), dtype=np.int32)
    print(f"onnx  encode: {codes_onnx.shape}")

    manifest_rows = runtime.list_builtin_voices()
    voice_row = next((item for item in manifest_rows if item["voice"] == args.voice), None)
    if voice_row is not None:
        codes_manifest = np.asarray(voice_row["prompt_audio_codes"], dtype=np.int32)
        print(f"manifest '{args.voice}': {codes_manifest.shape}")

        def compare(name, a, b):
            rows = min(a.shape[0], b.shape[0])
            same = float((a[:rows] == b[:rows]).mean()) * 100.0
            print(f"  {name}: rows {a.shape[0]} vs {b.shape[0]}, element match {same:.1f}% (first {rows} rows)")

        compare("torch vs manifest", codes_torch, codes_manifest)
        compare("onnx  vs manifest", codes_onnx, codes_manifest)
    compare("torch vs onnx", codes_torch, codes_onnx)

    # Decode both code sets with the ONNX codec and compare the resulting waveforms:
    # a healthy codec round trip keeps the two waveforms close.
    def decode(codes):
        channel_arrays, length = runtime.decode_full_audio(codes.tolist())
        return np.stack([np.asarray(channel[:length], dtype=np.float32) for channel in channel_arrays], axis=1)

    wave_t = decode(codes_torch)
    wave_o = decode(codes_onnx)
    samples = min(wave_t.shape[0], wave_o.shape[0])
    a, b = wave_t[:samples], wave_o[:samples]
    noise = float(np.mean((a - b) ** 2))
    signal = float(np.mean(a**2)) + 1e-12
    print(f"decoded waveforms: samples={samples} ({samples / sample_rate:.2f}s) "
          f"snr(torch_codes vs onnx_codes)={10 * np.log10(signal / noise):.2f} dB")
    print(f"  rms torch_codes={np.sqrt(np.mean(a**2)):.4f} rms onnx_codes={np.sqrt(np.mean(b**2)):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
