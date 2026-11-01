"""Rebuild the *old* (buggy) C++ prompt from a fixed one, for A/B listening tests.

The old ``prompt.cpp`` template had two defects versus the upstream builder:
  * it dropped the leading ``<|im_start|>`` (id 4), and
  * it kept a stray ``None`` (``505, 587``) after "- Text:", so the model read
    "None" aloud before every sentence.

This tool takes a correct prompt (``tools/make_prompt.py``) and re-inserts both
defects so the old behaviour can be reproduced without rebuilding the engine.

    python tools/make_legacy_prompt.py --in runs/cpp/prompt_fixed.txt --out runs/promptfix/prompt_legacy.txt --voice-frames 98
"""

from __future__ import annotations

import argparse
from pathlib import Path

STRAY_NONE_IDS = (505, 587)  # "▁N" + "one"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--voice-frames", type=int, required=True)
    parser.add_argument("--pad", type=int, default=1024)
    args = parser.parse_args()

    rows = [
        [int(v) for v in line.split()]
        for line in Path(args.src).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if rows[0][0] != 4:
        raise SystemExit("expected the fixed prompt to start with <|im_start|> (4)")
    body = rows[1:]  # drop <|im_start|>
    text_start = 12 + 1 + args.voice_frames + 1 + 56
    if text_start >= len(body):
        raise SystemExit(f"prompt too short for {args.voice_frames} voice frames")
    stray = []
    for token in STRAY_NONE_IDS:
        row = [args.pad] * len(rows[0])
        row[0] = token
        stray.append(row)
    legacy = body[:text_start] + stray + body[text_start:]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(" ".join(str(v) for v in row) for row in legacy) + "\n", encoding="utf-8")
    print(f"legacy prompt rows={len(legacy)} (stray None at {text_start}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
