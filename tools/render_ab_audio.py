"""Render one sentence with the PyTorch reference and the ONNX runtime, side by side.

Both backends receive the identical prompt rows (the checkpoint's own template, which
starts with ``<|im_start|>``), so greedy frames must match exactly; sampled frames only
differ because each backend draws its own random numbers.

Outputs land in ``runs/ab`` as wav files plus ``*_frames.txt`` dumps.

Usage:
    python tools/render_ab_audio.py --text "今天天气不错。"
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _torch_common import ONNX_TTS_DIR, REPO_DIR, RUNS_DIR, encode_reference, load_torch  # noqa: E402

OUT_DIR = RUNS_DIR / "ab"


def write_wav(path: Path, channels: list[np.ndarray], sample_rate: int) -> None:
    length = min(len(channel) for channel in channels)
    pcm = np.stack([np.asarray(channel[:length], dtype=np.float32) for channel in channels], axis=1)
    pcm16 = np.round(np.clip(pcm, -1.0, 1.0) * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(pcm16.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


def dump_frames(path: Path, frames: list[list[int]]) -> None:
    path.write_text("\n".join(" ".join(str(int(v)) for v in row) for row in frames), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="今天天气不错。")
    parser.add_argument("--prompt-audio", default=str(REPO_DIR / "assets" / "audio" / "zh_1.wav"))
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch, model, tokenizer, codec = load_torch(attn_implementation="sdpa")
    prompt_codes, sample_rate, _channels = encode_reference(model, codec, torch, args.prompt_audio)
    print(f"prompt codes: {prompt_codes.shape}")

    input_ids, mask = model.build_inference_input_ids(
        text=args.text,
        text_tokenizer=tokenizer,
        mode="voice_clone",
        prompt_audio_codes=torch.from_numpy(prompt_codes),
        device="cpu",
    )
    dump_frames(OUT_DIR / "prompt_rows.txt", input_ids[0].tolist())
    print(f"prompt rows: {tuple(input_ids.shape)} -> {OUT_DIR / 'prompt_rows.txt'}")

    sys.path.insert(0, str(REPO_DIR))
    from onnx_tts_runtime import OnnxTtsRuntime

    runtime = OnnxTtsRuntime(model_dir=ONNX_TTS_DIR, thread_count=4, do_sample=True)
    runtime.manifest["builtin_voices"][0]["prompt_audio_codes"] = prompt_codes.tolist()
    voice = runtime.manifest["builtin_voices"][0]["voice"]
    im_start = int(runtime.manifest["tts_config"]["im_start_token_id"])
    manifest_prefix = list(runtime.manifest["prompt_templates"]["user_prompt_prefix_token_ids"])

    def run_torch(sample: bool) -> list[list[int]]:
        torch.manual_seed(args.seed)
        with torch.no_grad():
            frames = model.generate(
                input_ids=input_ids,
                attention_mask=mask,
                max_new_frames=args.frames,
                do_sample=sample,
                text_temperature=1.0,
                text_top_p=1.0,
                text_top_k=50,
                audio_temperature=0.8,
                audio_top_p=0.95,
                audio_top_k=25,
                audio_repetition_penalty=1.2,
                return_dict_in_generate=True,
            ).audio_token_ids[0].tolist()
        return [[int(v) for v in row] for row in frames]

    def run_onnx(sample: bool, prefix: list[int]) -> list[list[int]]:
        runtime.manifest["prompt_templates"]["user_prompt_prefix_token_ids"] = prefix
        result = runtime.synthesize(
            text=args.text,
            voice=voice,
            output_audio_path=str(OUT_DIR / "_scratch.wav"),
            sample_mode="fixed" if sample else "greedy",
            do_sample=sample,
            streaming=False,
            max_new_frames=args.frames,
            enable_wetext=False,
            seed=args.seed,
        )
        return [[int(v) for v in row] for row in np.asarray(result["audio_token_ids"]).tolist()]

    jobs: list[tuple[str, list[list[int]]]] = []
    torch_greedy = run_torch(sample=False)
    jobs.append(("torch_greedy", torch_greedy))
    jobs.append((f"torch_sample_s{args.seed}", run_torch(sample=True)))
    onnx_greedy = run_onnx(sample=False, prefix=[im_start] + manifest_prefix)
    jobs.append(("onnx_greedy", onnx_greedy))
    jobs.append((f"onnx_sample_s{args.seed}", run_onnx(sample=True, prefix=[im_start] + manifest_prefix)))
    jobs.append(("onnx_greedy_manifest_prefix", run_onnx(sample=False, prefix=manifest_prefix)))
    runtime.manifest["prompt_templates"]["user_prompt_prefix_token_ids"] = [im_start] + manifest_prefix

    print()
    rows = min(len(torch_greedy), len(onnx_greedy))
    if rows:
        a = np.asarray(torch_greedy[:rows], dtype=np.int32)
        b = np.asarray(onnx_greedy[:rows], dtype=np.int32)
        match = (a == b).all(axis=1)
        print(f"greedy torch vs onnx: frames {len(torch_greedy)} vs {len(onnx_greedy)}, "
              f"identical rows {int(match.sum())}/{rows}, first divergence "
              f"{int(np.argmin(match)) if not match.all() else -1}")

    for name, frames in jobs:
        if not frames:
            print(f"  {name:32s} frames=  0  (generation stopped immediately)")
            continue
        print(f"  {name:32s} frames={len(frames):3d}  {len(frames) * 0.08:5.2f}s")
        channel_arrays, _length = runtime.decode_full_audio(frames)
        write_wav(OUT_DIR / f"{name}.wav", channel_arrays, sample_rate)
        dump_frames(OUT_DIR / f"{name}_frames.txt", frames)

    scratch = OUT_DIR / "_scratch.wav"
    if scratch.exists():
        scratch.unlink()
    print(f"\nwavs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
