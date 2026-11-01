"""生成 C++ 引擎的 prompt 文件（每行 17 个整数：text_id + 16 个 code）。

行结构与 ONNX 运行时完全一致：
    [prompt 前缀 token] + [参考音频 code 行] + [audio_end / 模板 / 正文 / assistant 前缀 / audio_start]

用法：
    python tools/make_prompt.py --text "..." --voice Junhao --out runs/cpp/prompt.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm

MODELS = Path(r"D:\workplace\audio\github")
TTS_DIR = MODELS / "MOSS-TTS-Nano-100M-ONNX"


def load_manifest() -> dict:
    return json.loads((TTS_DIR / "browser_poc_manifest.json").read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    """与 enable_wetext=False 时 ONNX 运行时一致：只走 robust 清洗。"""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "moss_tts_nano_onnx"))
    from tts_robust_normalizer_single_script import normalize_tts_text

    return normalize_tts_text(text)


def build_rows(text: str, voice: str, codes_path: str | None = None) -> tuple[list[list[int]], list[int]]:
    manifest = load_manifest()
    cfg = manifest["tts_config"]
    n_vq = int(cfg["n_vq"])
    pad = int(cfg["audio_pad_token_id"])

    if codes_path:
        # 任意音色档案：每行 16 个整数（frame-major）
        codes = [
            [int(v) for v in line.split()]
            for line in Path(codes_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in codes:
            if len(row) != n_vq:
                raise SystemExit(f"codes 每行需要 {n_vq} 个整数，实际 {len(row)}")
    else:
        voice_row = next((v for v in manifest["builtin_voices"] if v["voice"] == voice), None)
        if voice_row is None:
            raise SystemExit(f"找不到内置音色 {voice}；可用: {[v['voice'] for v in manifest['builtin_voices']]}")
        codes = voice_row["prompt_audio_codes"]

    sp = spm.SentencePieceProcessor(model_file=str(TTS_DIR / "tokenizer.model"))
    prepared = normalize_text(text)
    text_ids = [int(t) for t in sp.encode(prepared, out_type=int)]

    # 上游 builder 的第一个 token 是 <|im_start|>（id 4），manifest 的模板漏了它
    prefix = [
        int(cfg["im_start_token_id"]),
        *manifest["prompt_templates"]["user_prompt_prefix_token_ids"],
        int(cfg["audio_start_token_id"]),
    ]
    suffix = [
        int(cfg["audio_end_token_id"]),
        *manifest["prompt_templates"]["user_prompt_after_reference_token_ids"],
        *text_ids,
        *manifest["prompt_templates"]["assistant_prompt_prefix_token_ids"],
        int(cfg["audio_start_token_id"]),
    ]

    rows: list[list[int]] = []
    for token in prefix:
        row = [pad] * (n_vq + 1)
        row[0] = int(token)
        rows.append(row)
    for code_row in codes:
        row = [pad] * (n_vq + 1)
        row[0] = int(cfg["audio_user_slot_token_id"])
        for i, code in enumerate(code_row[:n_vq]):
            row[i + 1] = int(code)
        rows.append(row)
    for token in suffix:
        row = [pad] * (n_vq + 1)
        row[0] = int(token)
        rows.append(row)

    return rows, text_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="Junhao")
    parser.add_argument("--codes", default=None, help="音色档案 codes.txt（每行 16 个整数）")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows, text_ids = build_rows(args.text, args.voice, args.codes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(" ".join(str(v) for v in row) for row in rows) + "\n", encoding="utf-8")
    (out.parent / "prompt_meta.json").write_text(
        json.dumps({"text": args.text, "voice": args.voice, "codes": args.codes, "rows": len(rows), "text_tokens": len(text_ids)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out), "rows": len(rows), "text_tokens": len(text_ids)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
