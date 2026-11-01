"""Probe the first generated step of the original PyTorch checkpoint.

Prints the assistant decision logits (slot id 9 vs end id 7) for
  * the PyTorch prompt builder (with <|im_start|>), and
  * the browser ONNX manifest rows (same rows minus the leading <|im_start|>),
so we can see which prompt the checkpoint actually wants.

Usage:
    python tools/torch_probe_step0.py --text "今天天气不错。"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torch_common import REPO_DIR, load_torch  # noqa: E402

import torch  # noqa: E402


def load_model():
    _torch, model, text_tokenizer, codec = load_torch(attn_implementation="sdpa")
    return model, text_tokenizer, codec


@torch.no_grad()
def assistant_logits(model, input_ids, attention_mask):
    embeds = model._build_inputs_embeds(input_ids)
    out = model.transformer(
        input_ids=None,
        past_key_values=None,
        attention_mask=attention_mask,
        position_ids=None,
        inputs_embeds=embeds,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cu_seqlens=None,
        num_sequences=None,
    )
    hidden = out.last_hidden_state[:, -1, :].to(dtype=model.local_transformer.ln_f.weight.dtype)
    local = model._decode_local_last_hidden_state(hidden.unsqueeze(1))
    logits = model.text_lm_head(local)[0].float()
    return logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="今天天气不错。")
    parser.add_argument("--prompt-audio", default=str(REPO_DIR / "assets" / "audio" / "zh_1.wav"))
    args = parser.parse_args()

    model, text_tokenizer, codec = load_model()
    sample_rate = int(model._resolve_audio_tokenizer_sample_rate(codec))
    channels = int(model._resolve_audio_tokenizer_channels(codec))
    waveform, sr = model._load_reference_audio(Path(args.prompt_audio), sample_rate, channels)
    encoded = model._call_audio_encode(audio_tokenizer=codec, waveform=waveform, sample_rate=sr)
    prompt_codes = model._normalize_audio_codes(encoded)
    print(f"prompt_codes={tuple(prompt_codes.shape)}")

    text_ids = model._encode_text(text_tokenizer, args.text)
    print(f"text={args.text!r} text_ids={text_ids} pieces={[text_tokenizer.convert_ids_to_tokens(i) for i in text_ids]}")

    input_ids, mask = model.build_inference_input_ids(
        text=args.text,
        text_tokenizer=text_tokenizer,
        mode="voice_clone",
        prompt_audio_codes=prompt_codes,
        device="cpu",
    )
    print(f"pytorch prompt rows={input_ids.shape[1]}")

    logits = assistant_logits(model, input_ids, mask)
    slot, end = int(model.config.audio_assistant_slot_token_id), int(model.config.audio_end_token_id)
    print(f"  WITH im_start   : logit[slot]={logits[slot]:+.3f} logit[end]={logits[end]:+.3f} "
          f"diff={logits[slot] - logits[end]:+.3f} -> {'CONTINUE' if logits[slot] >= logits[end] else 'STOP'}")
    top = torch.topk(logits, 5)
    print(f"  top5 ids={[int(i) for i in top.indices]} values={[round(float(v), 2) for v in top.values]}")

    stripped_ids = input_ids[:, 1:, :]
    stripped_mask = torch.ones(stripped_ids.shape[:2], dtype=torch.bool)
    logits2 = assistant_logits(model, stripped_ids, stripped_mask)
    print(f"  WITHOUT im_start: logit[slot]={logits2[slot]:+.3f} logit[end]={logits2[end]:+.3f} "
          f"diff={logits2[slot] - logits2[end]:+.3f} -> {'CONTINUE' if logits2[slot] >= logits2[end] else 'STOP'}")
    top2 = torch.topk(logits2, 5)
    print(f"  top5 ids={[int(i) for i in top2.indices]} values={[round(float(v), 2) for v in top2.values]}")

    frames = model.generate(
        input_ids=input_ids,
        attention_mask=mask,
        max_new_frames=12,
        do_sample=False,
        return_dict_in_generate=True,
    ).audio_token_ids
    print(f"greedy 12-frame probe -> shape={tuple(frames.shape)}")
    for row in frames[0].tolist():
        print("   ", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
