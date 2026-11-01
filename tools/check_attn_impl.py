"""Check whether the checkpoint's own attention backends agree with the ONNX export.

``gpt2_decoder.py`` implements eager / sdpa / flash attention. If ``eager`` matches
the ONNX prefill graph while ``sdpa`` does not, then the CPU sdpa path is the broken
piece, not the ONNX export.

Usage:
    python tools/check_attn_impl.py --text "今天天气不错。"
"""

from __future__ import annotations

import argparse
import sys
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torch_common import REPO_DIR, encode_reference, load_torch, prefill_last_hidden  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="今天天气不错。")
    parser.add_argument("--prompt-audio", default=str(REPO_DIR / "assets" / "audio" / "zh_1.wav"))
    parser.add_argument("--drop-im-start", action="store_true")
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
    if args.drop_im_start:
        rows = rows[1:]
    print(f"rows={rows.shape}")

    hidden_eager, logits_eager = prefill_last_hidden(model, torch, rows)
    model._set_attention_implementation("sdpa")
    hidden_sdpa, logits_sdpa = prefill_last_hidden(model, torch, rows)
    model._set_attention_implementation("eager")

    sys.path.insert(0, str(REPO_DIR))
    from ort_cpu_runtime import OrtCpuRuntime, _flatten2d_int32, _flatten3d_int32

    from _torch_common import ONNX_TTS_DIR

    runtime = OrtCpuRuntime(model_dir=ONNX_TTS_DIR, do_sample=False)
    ids, dims = _flatten3d_int32([rows.tolist()])
    mask, mask_dims = _flatten2d_int32([[1] * rows.shape[0]])
    outputs = runtime.sessions["prefill"].run(
        None, {"input_ids": ids.reshape(dims), "attention_mask": mask.reshape(mask_dims)}
    )
    named = dict(zip([o.name for o in runtime.sessions["prefill"].get_outputs()], outputs, strict=True))
    hidden_onnx = named["global_hidden"][:, -1, :].astype(np.float32)[0]
    logits_onnx = runtime.run_local_cached_step(
        named["global_hidden"][:, -1, :].astype(np.float32),
        text_token_id=0,
        audio_token_id=0,
        channel_index=0,
        step_type=0,
        past_valid_lengths=0,
        local_past_by_name=runtime.create_empty_local_cached_past(),
    )[0]

    def compare(name_a, a, name_b, b):
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
        print(f"  hidden {name_a} vs {name_b}: cosine={cosine:.6f} rel_l2={rel:.4f}")

    print("global prefill hidden state comparison:")
    compare("eager", hidden_eager, "sdpa", hidden_sdpa)
    compare("eager", hidden_eager, "onnx", hidden_onnx)
    compare("sdpa", hidden_sdpa, "onnx", hidden_onnx)

    slot, end = int(model.config.audio_assistant_slot_token_id), int(model.config.audio_end_token_id)
    print("assistant logits [slot, end]:")
    for name, logits in (("eager", logits_eager), ("sdpa", logits_sdpa), ("onnx", logits_onnx)):
        print(f"  {name:6s} slot={float(logits[slot]):+.3f} end={float(logits[end]):+.3f} "
              f"top5={np.argsort(-logits)[:5].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
