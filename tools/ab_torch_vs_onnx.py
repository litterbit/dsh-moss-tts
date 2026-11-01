"""Compare the PyTorch checkpoint against the official ONNX runtime on identical input.

注意：两边用同一个参考音频 codes，但 prompt 模板不同（PyTorch ``build_inference_input_ids``
带前导 ``<|im_start|>``，ONNX manifest 的 ``user_prompt_prefix_token_ids`` 不带）。
因此帧级对比只作参考，不是正确性判据；模型数学一致性看 ``ab_prefill_logits.py``，
帧 0 硬校验看 ``check_torch_frame0.py``。

Usage:
    python tools/ab_torch_vs_onnx.py --text "今天天气不错。" --frames 16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torch_common import ONNX_TTS_DIR, REPO_DIR, RUNS_DIR, load_torch  # noqa: E402

OUT_DIR = RUNS_DIR / "official"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="今天天气不错。")
    parser.add_argument("--prompt-audio", default=str(REPO_DIR / "assets" / "audio" / "zh_1.wav"))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--voice", default="Junhao")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch, model, text_tokenizer, codec = load_torch()

    sample_rate = int(model._resolve_audio_tokenizer_sample_rate(codec))
    channels = int(model._resolve_audio_tokenizer_channels(codec))
    waveform, sr = model._load_reference_audio(Path(args.prompt_audio), sample_rate, channels)
    with torch.no_grad():
        encoded = model._call_audio_encode(audio_tokenizer=codec, waveform=waveform, sample_rate=sr)
    prompt_codes = model._normalize_audio_codes(encoded).numpy().astype(np.int32)
    print(f"reference codes: {prompt_codes.shape}")

    text_ids = model._encode_text(text_tokenizer, args.text)
    print(f"text ids: {text_ids}")

    with torch.no_grad():
        torch_rows, torch_mask = model.build_inference_input_ids(
            text=args.text,
            text_tokenizer=text_tokenizer,
            mode="voice_clone",
            prompt_audio_codes=torch.from_numpy(prompt_codes),
            device="cpu",
        )
        frames_torch = model.generate(
            input_ids=torch_rows,
            attention_mask=torch_mask,
            max_new_frames=args.frames,
            do_sample=False,
            return_dict_in_generate=True,
        ).audio_token_ids[0].tolist()
    print(f"torch rows={torch_rows.shape[1]} frames={len(frames_torch)} first={frames_torch[0] if frames_torch else []}")

    sys.path.insert(0, str(REPO_DIR))
    from onnx_tts_runtime import OnnxTtsRuntime

    runtime = OnnxTtsRuntime(model_dir=ONNX_TTS_DIR, thread_count=4, max_new_frames=args.frames, do_sample=False)
    runtime.manifest["builtin_voices"][0]["prompt_audio_codes"] = prompt_codes.tolist()
    result = runtime.synthesize(
        text=args.text,
        voice=runtime.manifest["builtin_voices"][0]["voice"],
        output_audio_path=str(OUT_DIR / "ab_onnx_same_prompt.wav"),
        sample_mode="greedy",
        do_sample=False,
        streaming=False,
        max_new_frames=args.frames,
        enable_wetext=False,
    )
    frames_onnx = np.asarray(result["audio_token_ids"], dtype=np.int32).tolist()
    print(f"onnx  frames={len(frames_onnx)} first={frames_onnx[0] if frames_onnx else []}")

    onnx_rows = runtime.build_voice_clone_request_rows(prompt_codes.tolist(), text_ids)["inputIds"]
    torch_rows_list = torch_rows[0].tolist()
    same_prompt = len(onnx_rows) == len(torch_rows_list) and all(
        int(a[0]) == int(b[0]) for a, b in zip(onnx_rows, torch_rows_list)
    )
    print(f"onnx rows={len(onnx_rows)} prompt_rows_identical={same_prompt}")
    if not same_prompt:
        print(
            "NOTE: 两个后端的 prompt 构造不同（PyTorch builder 带前导 <|im_start|>，ONNX manifest 不带），"
            "所以下面的帧级差异不能当作 torch 正确性判据。\n"
            "      模型数学一致性看 tools/ab_prefill_logits.py（prefill hidden + assistant logits），\n"
            "      帧 0 硬校验看 tools/check_torch_frame0.py（同一份 rows 喂 C++/ONNX 验证过的 prompt）。"
        )

    n = min(len(frames_torch), len(frames_onnx))
    if n == 0:
        print("no frames to compare")
        return 1
    a = np.asarray(frames_torch[:n], dtype=np.int32)
    b = np.asarray(frames_onnx[:n], dtype=np.int32)
    row_match = (a == b).all(axis=1)
    first_bad = int(np.argmin(row_match)) if not row_match.all() else -1
    elem_match = float((a == b).mean())
    print(f"\nrow-identical {int(row_match.sum())}/{n}  element-match {elem_match * 100:.1f}%  first divergent frame {first_bad}")
    for index in range(min(n, 4)):
        print(f"  f{index} torch={a[index].tolist()}")
        print(f"  f{index} onnx ={b[index].tolist()}")
    (OUT_DIR / "ab_torch_frames.txt").write_text(
        "\n".join(" ".join(str(v) for v in row) for row in frames_torch), encoding="utf-8"
    )
    (OUT_DIR / "ab_onnx_frames.txt").write_text(
        "\n".join(" ".join(str(v) for v in row) for row in frames_onnx), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
