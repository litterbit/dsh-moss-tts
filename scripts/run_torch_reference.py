"""用原始 PyTorch checkpoint 跑官方推理（本地权重、零 HF 缓存、零联网）。

    python scripts/run_torch_reference.py --greedy --text "今天天气不错。"
    python scripts/run_torch_reference.py --seed 0 --text "..." --prompt-audio ref.wav

模型代码直接从 ``MOSS-TTS-Nano-100M`` 目录 import，权重走 ``load_state_dict``，
因此 transformers 5.x 不会把非持久 buffer（RoPE 的 inv_freq）置成未初始化值。
详见 ``src/dsh_moss_tts/torch_ref.py`` 与 ``doc/torch_reference.md``。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_moss_tts import torch_ref  # noqa: E402
from dsh_moss_tts.paths import CODEC_CKPT_DIR, TTS_CKPT_DIR, TTS_REPO_DIR, run_dir  # noqa: E402

DEFAULT_PROMPT_AUDIO = TTS_REPO_DIR / "assets" / "audio" / "zh_1.wav"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", default="今天天气不错。")
    parser.add_argument("--prompt-audio", default=str(DEFAULT_PROMPT_AUDIO))
    parser.add_argument("--mode", default="voice_clone", choices=("voice_clone", "continuation"))
    parser.add_argument("--frames", type=int, default=200, help="max_new_frames")
    parser.add_argument("--nq", type=int, default=None, help="只用前 nq 层 RVQ（默认全部 16）")
    parser.add_argument("--greedy", action="store_true", help="贪心解码；默认采样")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out-dir", default=None, help="默认 runs/official/")
    parser.add_argument("--ckpt", default=str(TTS_CKPT_DIR))
    parser.add_argument("--codec", default=str(CODEC_CKPT_DIR))
    parser.add_argument("--attn", default="sdpa", choices=("sdpa", "eager"))
    parser.add_argument("--voice-clone-max-text-tokens", type=int, default=75, help="<=0 关闭长文本分块")
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-top-p", type=float, default=0.95)
    parser.add_argument("--audio-top-k", type=int, default=25)
    parser.add_argument("--audio-repetition-penalty", type=float, default=1.2)
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument("--text-top-p", type=float, default=1.0)
    parser.add_argument("--text-top-k", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir("official")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("greedy" if args.greedy else f"seed{args.seed}")

    started = time.perf_counter()
    torch, model, tokenizer, codec = torch_ref.load_all(
        args.ckpt,
        args.codec,
        attn_implementation=args.attn,
    )
    load_seconds = time.perf_counter() - started
    print(f"loaded in {load_seconds:.2f}s: {type(model).__name__} / {type(codec).__name__}", flush=True)

    torch.manual_seed(args.seed)
    wav_path = out_dir / f"torch_{tag}.wav"
    started = time.perf_counter()
    result = model.inference(
        text=args.text,
        output_audio_path=str(wav_path),
        mode=args.mode,
        prompt_audio_path=args.prompt_audio,
        text_tokenizer=tokenizer,
        audio_tokenizer=codec,
        device="cpu",
        nq=args.nq,
        max_new_frames=args.frames,
        do_sample=not args.greedy,
        text_temperature=args.text_temperature,
        text_top_p=args.text_top_p,
        text_top_k=args.text_top_k,
        audio_temperature=args.audio_temperature,
        audio_top_p=args.audio_top_p,
        audio_top_k=args.audio_top_k,
        audio_repetition_penalty=args.audio_repetition_penalty,
        voice_clone_max_text_tokens=args.voice_clone_max_text_tokens,
    )
    synth_seconds = time.perf_counter() - started

    frames = result["audio_token_ids"]
    waveform = result["waveform"].detach().float()
    codes_path = out_dir / f"torch_{tag}_frames.txt"
    rows = frames.tolist()
    codes_path.write_text("\n".join(" ".join(str(int(v)) for v in row) for row in rows), encoding="utf-8")

    peak = float(waveform.abs().max())
    rms = float(waveform.pow(2).mean().sqrt())
    sample_rate = int(result["sample_rate"])
    duration = waveform.shape[-1] / sample_rate if sample_rate else 0.0

    print(f"text={args.text!r} mode={args.mode} frames={frames.shape[0]}")
    print(f"first_codes={rows[0] if rows else []}")
    print(f"wav={wav_path}")
    print(
        f"sample_rate={sample_rate} channels={waveform.shape[0]} duration={duration:.3f}s "
        f"peak={peak:.4f} rms={rms:.4f} synth={synth_seconds:.2f}s"
    )
    print(f"codes={codes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
