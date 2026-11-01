"""比较 C++ 解码输出与 ONNX decode_full 的音频（SNR）。

用法：
    python tools/compare_audio.py --cpp runs/ablation/cpp_audio.f32 --codes runs/cpp/codes_q8x4_67.txt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ONNX_DIR = Path(r"D:\workplace\audio\github\MOSS-Audio-Tokenizer-Nano-ONNX")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", required=True, help="交织立体声 float32 原始文件")
    parser.add_argument("--codes", required=True)
    parser.add_argument("--save-wav", default=None)
    args = parser.parse_args()

    inter = np.fromfile(args.cpp, dtype=np.float32)
    mine = np.stack([inter[0::2], inter[1::2]], axis=0)

    rows = [[int(x) for x in line.split()] for line in Path(args.codes).read_text().splitlines() if line.strip()]
    codes = np.array(rows, dtype=np.int32)
    frames = codes.shape[0]

    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_DIR / "moss_audio_tokenizer_decode_full.onnx"))
    audio, lengths = sess.run(
        None,
        {"audio_codes": codes[None], "audio_code_lengths": np.array([frames], dtype=np.int32)},
    )
    n = int(np.asarray(lengths).reshape(-1)[0])
    ref = audio[0, :, :n]

    m = min(mine.shape[-1], ref.shape[-1])
    a = ref[:, :m].astype(np.float64)
    b = mine[:, :m].astype(np.float64)
    noise = a - b
    snr = 10.0 * math.log10(float((a * a).sum()) / max(float((noise * noise).sum()), 1e-20))

    result = {
        "frames": frames,
        "cpp_shape": list(mine.shape),
        "onnx_shape": list(ref.shape),
        "snr_db": round(snr, 2),
        "peak_cpp": round(float(np.abs(mine).max()), 4),
        "peak_onnx": round(float(np.abs(ref).max()), 4),
        "rms_cpp": round(float(np.sqrt((mine.astype(np.float64) ** 2).mean())), 5),
    }
    print(json.dumps(result, ensure_ascii=False))

    if args.save_wav:
        import wave

        pcm = np.clip(mine.T, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2")
        with wave.open(args.save_wav, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm16.tobytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
