"""把 manifest 里的内置音色导出成 .mossclone 档案（音色码 + 名字）。

用法：
    python tools/export_voices.py --manifest --out-dir weights/voices
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

MAGIC = b"MSCL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    n_vq = int(manifest["tts_config"]["n_vq"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for voice in manifest["builtin_voices"]:
        name = voice["voice"]
        codes = voice["prompt_audio_codes"]
        blob = bytearray()
        blob += MAGIC
        blob += struct.pack("<IIIII", 1, n_vq, len(codes), 48000, len(name.encode("utf-8")))
        blob += name.encode("utf-8")
        for row in codes:
            blob += struct.pack("<16h", *[int(v) for v in row[:n_vq]])
        path = out_dir / f"{name}.mossclone"
        path.write_bytes(bytes(blob))
        written.append({"voice": name, "frames": len(codes), "bytes": path.stat().st_size})

    print(json.dumps({"out_dir": str(out_dir), "count": len(written), "voices": written}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
