"""Verify that the checkpoint weights really land in the model built by transformers.

transformers 5.x is much newer than this checkpoint's remote code, so a silent
"missing/unexpected" weight report is the first thing to rule out before trusting
anything the PyTorch path produces.

Usage:
    python tools/check_weights_load.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torch_common import CKPT_DIR, load_torch  # noqa: E402


def main() -> int:
    state = torch.load(str(CKPT_DIR / "pytorch_model.bin"), map_location="cpu", weights_only=True)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    print(f"checkpoint tensors: {len(state)}")

    _torch, model, _tok, _codec = load_torch(attn_implementation="eager", with_codec=False)
    model_state = model.state_dict()
    print(f"model tensors: {len(model_state)}")

    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    print(f"missing in checkpoint: {len(missing)}")
    for name in missing[:20]:
        print(f"  missing {name} {tuple(model_state[name].shape)}")
    print(f"unexpected in checkpoint: {len(unexpected)}")
    for name in unexpected[:20]:
        print(f"  unexpected {name} {tuple(state[name].shape)}")

    mismatched = [
        name
        for name in sorted(set(state) & set(model_state))
        if tuple(state[name].shape) != tuple(model_state[name].shape)
    ]
    print(f"shape mismatches: {len(mismatched)}")
    for name in mismatched[:20]:
        print(f"  shape {name} ckpt={tuple(state[name].shape)} model={tuple(model_state[name].shape)}")

    probes = [
        "transformer.wte.weight",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.11.mlp.c_fc.weight",
        "text_lm_head.weight",
        "audio_embeddings.0.weight",
        "audio_lm_heads.0.weight",
    ]
    print("value probes (max abs diff between file and loaded model):")
    for name in probes:
        if name not in state or name not in model_state:
            print(f"  {name}: absent (ckpt={name in state} model={name in model_state})")
            continue
        a = state[name].float()
        b = model_state[name].float()
        diff = float((a - b).abs().max()) if a.shape == b.shape else float("nan")
        print(f"  {name}: maxdiff={diff:.6e} ckpt_absmean={float(a.abs().mean()):.4f} "
              f"model_absmean={float(b.abs().mean()):.4f}")

    index = {name: i for i, name in enumerate(model_state)}
    print(f"'audio_embeddings' present in model: "
          f"{[k for k in model_state if 'audio_embed' in k][:3]}")
    print(f"'audio_lm_heads' present in model: "
          f"{[k for k in model_state if 'audio_lm' in k][:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
