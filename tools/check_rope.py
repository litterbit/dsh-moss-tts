"""Sanity-check the checkpoint's rotary embedding under the installed transformers.

``inv_freq`` is a non-persistent buffer, so it is NOT part of pytorch_model.bin: it is
computed at construction time. If transformers 5.x changed how the remote code builds
GPT2RotaryEmbedding, the model silently runs with the wrong rotations and produces
noise while every weight still matches the file bit-for-bit.

Usage:
    python tools/check_rope.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torch_common import load_torch  # noqa: E402


def main() -> int:
    torch, model, _tokenizer, _codec = load_torch(attn_implementation="eager", with_codec=False)
    transformer = model.transformer
    print(f"position_embedding_type={getattr(transformer, 'position_embedding_type', None)}")
    attn = transformer.h[0].attn
    rotary = getattr(attn, "rotary_emb", None)
    print(f"rotary_emb type={type(rotary).__name__ if rotary is not None else None}")
    if rotary is not None:
        inv_freq = getattr(rotary, "inv_freq", None)
        if inv_freq is None:
            print("no inv_freq attribute")
        else:
            values = inv_freq.detach().float().cpu().numpy().reshape(-1)
            print(f"inv_freq[:8]={np.round(values[:8], 6).tolist()}")
            base = float(np.round(1.0 / values[0], 3))
            print(f"implied rope base={base}")
            head_dim = int(model.config.gpt2_config.n_embd // model.config.gpt2_config.n_head)
            expected = 10000.0 ** (-2.0 * np.arange(values.size) / head_dim)
            print(f"expected[0]={expected[0]:.6f} expected[-1]={expected[-1]:.6f}")
            print(f"max rel error vs 10000-base formula: {float(np.max(np.abs(values - expected) / expected)):.3e}")
        for name in ("dim", "head_dim", "base", "rope_base", "max_seq_len_cached"):
            if hasattr(rotary, name):
                print(f"  rotary_emb.{name}={getattr(rotary, name)}")

    with torch.no_grad():
        cos, sin = attn.rotary_emb(torch.arange(4).unsqueeze(0), device="cpu", dtype=torch.float32)
    cos, sin = cos[0], sin[0]  # [T, 1, head_dim]
    print(f"cos shape={tuple(cos.shape)} cos[t0,:6]={np.round(cos[0, 0, :6].numpy(), 4).tolist()}")
    print(f"sin shape={tuple(sin.shape)} sin[t1,:6]={np.round(sin[1, 0, :6].numpy(), 4).tolist()}")
    print(f"cos[t3,:6]={np.round(cos[3, 0, :6].numpy(), 4).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
