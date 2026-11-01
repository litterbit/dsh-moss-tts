"""Compare the local decoder of the checkpoint against the ONNX local graphs.

Both backends are fed the *same* global hidden state, so this isolates the per-frame
local transformer + per-channel heads + sampling.

Usage:
    python tools/check_local_decoder.py --text "今天天气不错。"
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
    hidden, _logits = prefill_last_hidden(model, torch, rows_np)
    hidden_t = torch.from_numpy(hidden).unsqueeze(0)

    # --- PyTorch: one full greedy frame, mirroring _iter_generation_events ------------
    slot_id = int(model.config.audio_assistant_slot_token_id)
    local_dtype = model.local_transformer.ln_f.weight.dtype
    local_inputs = hidden_t.unsqueeze(1).to(dtype=local_dtype)
    with torch.no_grad():
        local_hidden = model._decode_local_last_hidden_state(local_inputs)
        text_logits = model.text_lm_head(local_hidden)[0].float().numpy()
        frame_torch = []
        current = model.transformer.wte(torch.tensor([slot_id])).to(dtype=local_dtype)
        channel_logits = []
        for channel_index in range(int(model.config.n_vq)):
            local_inputs = torch.cat([local_inputs, current.unsqueeze(1)], dim=1)
            local_hidden = model._decode_local_last_hidden_state(local_inputs)
            logits = model.audio_lm_heads[channel_index](local_hidden)[0].float().numpy()
            channel_logits.append(logits)
            token = int(np.argmax(logits))
            frame_torch.append(token)
            current = model.audio_embeddings[channel_index](torch.tensor([token])).to(dtype=local_dtype)

    # --- ONNX: greedy frame from the same hidden state -------------------------------
    sys.path.insert(0, str(REPO_DIR))
    from ort_cpu_runtime import OrtCpuRuntime

    runtime = OrtCpuRuntime(model_dir=ONNX_TTS_DIR, do_sample=False)
    n_vq = int(runtime.manifest["tts_config"]["n_vq"])
    print(f"available onnx sessions: {sorted(runtime.sessions)}")
    runtime.manifest["generation_defaults"]["max_new_frames"] = 1
    frames_onnx = runtime.generate_audio_frames(
        {"inputIds": rows_np.tolist(), "attentionMask": [[1] * rows_np.shape[0]]}
    )
    frame_onnx = [int(v) for v in frames_onnx[0]] if frames_onnx else []
    should_continue = bool(frames_onnx)
    print(f"continue (onnx)={should_continue}")
    print(f"frame torch={frame_torch}")
    print(f"frame onnx ={frame_onnx}")

    # per-channel logits comparison at channel 0 via the cached-step graph
    _text_logits, audio_logits, _past = runtime.run_local_cached_step(
        hidden.astype(np.float32).reshape(1, -1),
        text_token_id=slot_id,
        audio_token_id=0,
        channel_index=0,
        step_type=1,
        past_valid_lengths=1,
        local_past_by_name=runtime.create_empty_local_cached_past(),
    )
    onnx_ch0 = runtime.slice_audio_channel_logits(audio_logits, 0).astype(np.float32)
    d = float(np.max(np.abs(onnx_ch0 - channel_logits[0])))
    cos = float(np.dot(onnx_ch0, channel_logits[0]) /
                (np.linalg.norm(onnx_ch0) * np.linalg.norm(channel_logits[0]) + 1e-12))
    print(f"channel0 logits: maxdiff={d:.4f} cosine={cos:.6f} "
          f"argmax torch={int(np.argmax(channel_logits[0]))} onnx={int(np.argmax(onnx_ch0))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
