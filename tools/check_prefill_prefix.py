"""Find the first prompt length where the ONNX prefill stops matching the checkpoint.

Walks the voice-clone prompt row by row (text rows, then reference-audio rows, then
the suffix) and prints the cosine between the checkpoint's last hidden state and the
ONNX prefill graph's last hidden state for each prefix length.

Usage:
    python tools/check_prefill_prefix.py --text "今天天气不错。"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torch_common import (  # noqa: E402
    ONNX_TTS_DIR,
    REPO_DIR,
    encode_reference,
    load_torch,
    prefill_last_hidden,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="今天天气不错。")
    parser.add_argument("--prompt-audio", default=str(REPO_DIR / "assets" / "audio" / "zh_1.wav"))
    parser.add_argument("--step", type=int, default=6)
    args = parser.parse_args()

    torch, model, tokenizer, codec = load_torch(attn_implementation="eager")
    codes, _sr, _ch = encode_reference(model, codec, torch, args.prompt_audio)
    input_ids, _mask = model.build_inference_input_ids(
        text=args.text,
        text_tokenizer=tokenizer,
        mode="voice_clone",
        prompt_audio_codes=torch.from_numpy(codes),
        device="cpu",
    )
    rows = input_ids[0].numpy().astype(np.int32)
    total = int(rows.shape[0])
    print(f"total rows={total} (prefix 14 + audio {codes.shape[0]} + suffix {total - 14 - codes.shape[0]})")

    sys.path.insert(0, str(REPO_DIR))
    from ort_cpu_runtime import OrtCpuRuntime, _flatten2d_int32, _flatten3d_int32

    runtime = OrtCpuRuntime(model_dir=ONNX_TTS_DIR, do_sample=False)
    lengths = sorted({1, 2, 3, 4, 6, 8, 10, 12, 14, 20, 40, 60, 100, total})
    for length in lengths:
        if length > total:
            continue
        prefix = rows[:length]
        hidden_t, logits_t = prefill_last_hidden(model, torch, prefix)
        ids, dims = _flatten3d_int32([prefix.tolist()])
        mask, mask_dims = _flatten2d_int32([[1] * length])
        outputs = runtime.sessions["prefill"].run(
            None, {"input_ids": ids.reshape(dims), "attention_mask": mask.reshape(mask_dims)}
        )
        named = dict(zip([o.name for o in runtime.sessions["prefill"].get_outputs()], outputs, strict=True))
        hidden_o = named["global_hidden"][:, -1, :].astype(np.float32)[0]
        cosine = float(np.dot(hidden_t, hidden_o) / (np.linalg.norm(hidden_t) * np.linalg.norm(hidden_o) + 1e-12))
        rel = float(np.linalg.norm(hidden_t - hidden_o) / (np.linalg.norm(hidden_t) + 1e-12))
        print(f"  len={length:4d} cosine={cosine:.6f} rel_l2={rel:.4f} torch_norm={np.linalg.norm(hidden_t):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
