"""Compare the C++ engine's frame-0 logits against the PyTorch reference.

Reads the dumps produced by ``moss_llm --dump-logits/--dump-hidden`` and recomputes the
same logits with the checkpoint, so the first divergent channel is visible.

Usage:
    python tools/check_cpp_frame0.py --text "今天天气不错。"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torch_common import REPO_DIR, RUNS_DIR, encode_reference, load_torch, prefill_last_hidden  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="今天天气不错。")
    parser.add_argument("--prompt-audio", default=str(REPO_DIR / "assets" / "audio" / "zh_1.wav"))
    parser.add_argument("--cpp-logits", default=str(RUNS_DIR / "cpp" / "logits_f0.bin"))
    parser.add_argument("--cpp-hidden", default=str(RUNS_DIR / "cpp" / "hidden_f0.bin"))
    args = parser.parse_args()

    torch, model, tokenizer, codec = load_torch(attn_implementation="sdpa")
    codes, _sr, _ch = encode_reference(model, codec, torch, args.prompt_audio)
    rows, _mask = model.build_inference_input_ids(
        text=args.text,
        text_tokenizer=tokenizer,
        mode="voice_clone",
        prompt_audio_codes=torch.from_numpy(codes),
        device="cpu",
    )
    rows_np = rows[0].numpy().astype(np.int32)
    hidden, text_logits = prefill_last_hidden(model, torch, rows_np)

    cpp_hidden = np.fromfile(args.cpp_hidden, dtype=np.float32)
    if cpp_hidden.size == hidden.size:
        cosine = float(np.dot(hidden, cpp_hidden) / (np.linalg.norm(hidden) * np.linalg.norm(cpp_hidden) + 1e-12))
        print(f"prefill hidden: cosine={cosine:.6f} rel_l2={np.linalg.norm(hidden - cpp_hidden) / np.linalg.norm(hidden):.5f}")
    else:
        print(f"hidden size mismatch: cpp={cpp_hidden.size} torch={hidden.size}")

    slot_id = int(model.config.audio_assistant_slot_token_id)
    local_dtype = model.local_transformer.ln_f.weight.dtype
    local_inputs = torch.from_numpy(hidden).unsqueeze(0).unsqueeze(1).to(dtype=local_dtype)
    ref_logits: list[np.ndarray] = []
    frame: list[int] = []
    with torch.no_grad():
        local_hidden = model._decode_local_last_hidden_state(local_inputs)
        ref_logits.append(model.text_lm_head(local_hidden)[0].float().numpy())
        current = model.transformer.wte(torch.tensor([slot_id])).to(dtype=local_dtype)
        for channel in range(int(model.config.n_vq)):
            local_inputs = torch.cat([local_inputs, current.unsqueeze(1)], dim=1)
            local_hidden = model._decode_local_last_hidden_state(local_inputs)
            logits = model.audio_lm_heads[channel](local_hidden)[0].float().numpy()
            ref_logits.append(logits)
            token = int(np.argmax(logits))
            frame.append(token)
            current = model.audio_embeddings[channel](torch.tensor([token])).to(dtype=local_dtype)

    raw = np.fromfile(args.cpp_logits, dtype=np.float32)
    block = 1024
    if raw.size % block != 0:
        print(f"unexpected logits dump size {raw.size}")
        return 1
    cpp = raw.reshape(-1, block)
    print(f"cpp logits rows={cpp.shape[0]} (1 text + {cpp.shape[0] - 1} channels)")

    # moss_llm 只 dump 16 路音频头（不含 text 头）
    names = [f"ch{index}" for index in range(cpp.shape[0])]
    for index, name in enumerate(names):
        if index + 1 >= len(ref_logits) or index >= cpp.shape[0]:
            break
        a, b = ref_logits[index + 1], cpp[index]
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"  {name:5s} maxdiff={np.max(np.abs(a - b)):.4f} cosine={cosine:.6f} "
              f"argmax torch={int(np.argmax(a))} cpp={int(np.argmax(b))}")
    print(f"torch frame 0 = {frame}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
