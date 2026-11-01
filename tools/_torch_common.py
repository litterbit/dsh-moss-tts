"""Shared loader for the original PyTorch MOSS-TTS-Nano checkpoint.

The actual loading lives in ``src/dsh_moss_tts/torch_ref.py``: it imports the
checkpoint's own modules straight from the model directory, builds the model from
``config.json`` and calls ``load_state_dict``. That keeps non-persistent buffers
(notably every ``rotary_emb.inv_freq``) at the values produced by ``__init__``.

Do NOT switch this back to ``from_pretrained(..., trust_remote_code=True)``:
transformers 5.x materialises non-persistent buffers with ``torch.empty_like``,
which silently turns RoPE into the identity and makes the model emit noise while
every weight still loads bit-exactly. See ``doc/torch_reference.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dsh_moss_tts import torch_ref  # noqa: E402
from dsh_moss_tts.paths import (  # noqa: E402
    CODEC_CKPT_DIR,
    MODELS_DIR,
    RUNS_DIR,
    TTS_CKPT_DIR,
    TTS_REPO_DIR,
)

CKPT_DIR = TTS_CKPT_DIR
CODEC_DIR = CODEC_CKPT_DIR
REPO_DIR = TTS_REPO_DIR
ONNX_TTS_DIR = MODELS_DIR / "MOSS-TTS-Nano-100M-ONNX"


def load_torch(attn_implementation: str = "sdpa", with_codec: bool = True):
    """Return ``(torch, model, text_tokenizer, codec_or_None)`` on CPU float32."""
    return torch_ref.load_all(
        CKPT_DIR,
        CODEC_DIR,
        attn_implementation=attn_implementation,
        with_codec=with_codec,
    )


def encode_reference(model, codec, torch, prompt_audio: str | Path):
    """Encode a reference wav into prompt rows using the checkpoint's own codec path."""
    sample_rate = int(model._resolve_audio_tokenizer_sample_rate(codec))
    channels = int(model._resolve_audio_tokenizer_channels(codec))
    waveform, rate = model._load_reference_audio(Path(prompt_audio), sample_rate, channels)
    with torch.no_grad():
        encoded = model._call_audio_encode(audio_tokenizer=codec, waveform=waveform, sample_rate=rate)
    codes = model._normalize_audio_codes(encoded).numpy().astype("int32")
    return codes, sample_rate, channels


def prefill_last_hidden(model, torch, rows, attention_mask=None):
    """Run the global transformer over ``rows`` (T, 17) and return the last hidden state."""
    import numpy as np

    rows_t = torch.as_tensor(np.asarray(rows), dtype=torch.long).unsqueeze(0)
    length = int(rows_t.shape[1])
    mask = (
        torch.ones(1, length, dtype=torch.bool)
        if attention_mask is None
        else torch.as_tensor(attention_mask, dtype=torch.bool).unsqueeze(0)
    )
    with torch.no_grad():
        embeds = model._build_inputs_embeds(rows_t)
        out = model.transformer(
            input_ids=None,
            past_key_values=None,
            attention_mask=mask,
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
        local_hidden = model._decode_local_last_hidden_state(hidden.unsqueeze(1))
        text_logits = model.text_lm_head(local_hidden)[0].float().numpy()
    return hidden[0].numpy(), text_logits
