"""把 data/clone_wav 里的参考音频做成网页可选的克隆音色档案。

- 源文件是 MP3-in-WAV（fmt tag 0x55），先用 torchaudio 转成 PCM WAV；
- 再调 moss_clone.exe（发布包里的那个）生成 .mossclone。

用法：
    python tools/prepare_clone_voices.py [--src data/clone_wav] [--out web/voices]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import torchaudio

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "moss-tts-cpu-win64"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(ROOT / "data" / "clone_wav"))
    parser.add_argument("--out", default=str(ROOT / "web" / "voices"))
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exe = DIST / "moss_clone.exe"
    encoder = DIST / "models" / "encoder_f16.moss"
    if not exe.exists() or not encoder.exists():
        raise SystemExit(f"找不到发布包里的 moss_clone.exe / encoder：{DIST}")

    report = []
    for wav in sorted(src.glob("*.wav")):
        name = wav.stem
        pcm = out / f"{name}.src.wav"
        waveform, sample_rate = torchaudio.load(str(wav))
        torchaudio.save(str(pcm), waveform, sample_rate)
        profile = out / f"{name}.mossclone"
        proc = subprocess.run(
            [str(exe), "--encoder", str(encoder), "--in", str(pcm), "--out", str(profile), "--name", name,
             "--dump-codes", str(out / f"{name}.codes.txt")],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0:
            report.append({"voice": name, "ok": False, "err": (proc.stderr or proc.stdout)[-400:]})
            continue
        # 注意：C++ 打印的 JSON 里路径是单反斜杠（不合法），只按行提取数字字段
        stats = {k: float(v) for k, v in re.findall(r'"(\w+)":\s*(-?[\d.]+)', proc.stdout)}
        report.append({"voice": name, "ok": True, "frames": int(stats["frames"]),
                       "seconds": stats["seconds"], "encode_ms": stats["encode_ms"],
                       "kb": round(profile.stat().st_size / 1024, 1)})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(r["ok"] for r in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
