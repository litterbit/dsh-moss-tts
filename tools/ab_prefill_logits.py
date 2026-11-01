"""Locate where the ONNX graphs diverge from the PyTorch checkpoint.

Compares, on identical prompt rows:
  1. the prefill hidden state (last row), and
  2. the assistant decision logits produced by the local decoder at step 0.

Usage:
    python tools/ab_prefill_logits.py --text "今天天气不错。"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torch_common import ONNX_TTS_DIR, REPO_DIR, load_torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="今天天气不错。")
    parser.add_argument("--prompt-audio", default=str(REPO_DIR / "assets" / "audio" / "zh_1.wav"))
    parser.add_argument("--drop-im-start", action="store_true")
    args = parser.parse_args()

    torch, model, tokenizer, codec = load_torch()
    sr = int(model._resolve_audio_tokenizer_sample_rate(codec))
    ch = int(model._resolve_audio_tokenizer_channels(codec))
    waveform, rate = model._load_reference_audio(Path(args.prompt_audio), sr, ch)
    with torch.no_grad():
        codes = model._normalize_audio_codes(
            model._call_audio_encode(audio_tokenizer=codec, waveform=waveform, sample_rate=rate)
        ).numpy().astype(np.int32)
    text_ids = model._encode_text(tokenizer, args.text)

    input_ids, _mask = model.build_inference_input_ids(
        text=args.text,
        text_tokenizer=tokenizer,
        mode="voice_clone",
        prompt_audio_codes=torch.from_numpy(codes),
        device="cpu",
    )
    rows_np = input_ids[0].numpy().astype(np.int32)
    if args.drop_im_start:
        rows_np = rows_np[1:]
    print(f"rows={rows_np.shape}  (im_start dropped={args.drop_im_start})")
    rows_t = torch.from_numpy(rows_np).unsqueeze(0)

    with torch.no_grad():
        embeds = model._build_inputs_embeds(rows_t)
        out = model.transformer(
            input_ids=None,
            past_key_values=None,
            attention_mask=torch.ones(1, rows_np.shape[0], dtype=torch.bool),
            position_ids=None,
            inputs_embeds=embeds,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cu_seqlens=None,
            num_sequences=None,
        )
        hidden_t = out.last_hidden_state[:, -1, :].to(dtype=model.local_transformer.ln_f.weight.dtype)
        local_hidden = model._decode_local_last_hidden_state(hidden_t.unsqueeze(1))
        logits_t = model.text_lm_head(local_hidden)[0].float()

    sys.path.insert(0, str(REPO_DIR))
    from ort_cpu_runtime import OrtCpuRuntime, _flatten2d_int32, _flatten3d_int32

    rt = OrtCpuRuntime(model_dir=ONNX_TTS_DIR, do_sample=False)
    ids, dims = _flatten3d_int32([rows_np.tolist()])
    mask, mask_dims = _flatten2d_int32([[1] * rows_np.shape[0]])
    outs = rt.sessions["prefill"].run(
        None,
        {"input_ids": ids.reshape(dims), "attention_mask": mask.reshape(mask_dims)},
    )
    named = dict(zip([o.name for o in rt.sessions["prefill"].get_outputs()], outs, strict=True))
    hidden_o = named["global_hidden"][:, -1, :]
    logits_o = rt.run_local_cached_step(
        hidden_o.astype(np.float32),
        text_token_id=0,
        audio_token_id=0,
        channel_index=0,
        step_type=0,
        past_valid_lengths=0,
        local_past_by_name=rt.create_empty_local_cached_past(),
    )[0]

    ht = hidden_t.numpy()[0]
    ho = hidden_o[0]
    cosine = float(np.dot(ht, ho) / (np.linalg.norm(ht) * np.linalg.norm(ho) + 1e-12))
    print(f"prefill last hidden: cosine={cosine:.6f} rel_l2={np.linalg.norm(ht - ho) / np.linalg.norm(ht):.4f}")
    print(f"  torch[:6]={np.round(ht[:6], 4).tolist()}")
    print(f"  onnx [:6]={np.round(ho[:6], 4).tolist()}")

    slot, end = int(model.config.audio_assistant_slot_token_id), int(model.config.audio_end_token_id)
    print(f"assistant logits torch: slot={float(logits_t[slot]):+.3f} end={float(logits_t[end]):+.3f}")
    print(f"assistant logits onnx : slot={float(logits_o[slot]):+.3f} end={float(logits_o[end]):+.3f}")
    top_t = np.argsort(-logits_t.numpy())[:5]
    top_o = np.argsort(-logits_o)[:5]
    print(f"top5 torch={top_t.tolist()} {np.round(logits_t.numpy()[top_t], 2).tolist()}")
    print(f"top5 onnx ={top_o.tolist()} {np.round(logits_o[top_o], 2).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
