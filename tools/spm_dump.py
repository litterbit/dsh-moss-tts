"""把 SentencePiece 的 tokenizer.model 转成本项目自用的 .mossvocab（C++ 侧零依赖读取）。

包含：pieces（字节 + score + type）、特殊 id、normalizer 开关、precompiled charsmap（Darts trie + blob）。

用法：
    python tools/spm_dump.py --src tokenizer.model --out weights/tokenizer.mossvocab
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

MAGIC = b"MOSVOC1\0"


def read_varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return value, i


def fields(buf: bytes):
    i = 0
    while i < len(buf):
        key, i = read_varint(buf, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = read_varint(buf, i)
            yield f, wt, v
        elif wt == 2:
            n, i = read_varint(buf, i)
            yield f, wt, buf[i : i + n]
            i += n
        elif wt == 5:
            yield f, wt, buf[i : i + 4]
            i += 4
        elif wt == 1:
            yield f, wt, buf[i : i + 8]
            i += 8
        else:
            return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = Path(args.src).read_bytes()
    pieces: list[tuple[bytes, float, int]] = []
    trainer: dict[int, int] = {}
    norm: dict[int, object] = {}

    for f, wt, v in fields(data):
        if f == 1 and wt == 2:
            piece, score, ptype = b"", 0.0, 0
            for g, w2, v2 in fields(v):
                if g == 1 and w2 == 2:
                    piece = v2
                elif g == 2 and w2 == 5:
                    score = struct.unpack("<f", v2)[0]
                elif g == 3 and w2 == 0:
                    ptype = v2
            pieces.append((piece, score, ptype))
        elif f == 2 and wt == 2:
            for g, w2, v2 in fields(v):
                if w2 == 0:
                    trainer[g] = v2
        elif f == 3 and wt == 2:
            for g, w2, v2 in fields(v):
                norm[g] = v2

    special = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    unk = bos = eos = pad = -1
    for idx, (piece, _score, _ptype) in enumerate(pieces):
        name = piece.decode("utf-8", "replace")
        if name in special and special[name] == idx:
            if name == "<unk>":
                unk = idx
            elif name == "<s>":
                bos = idx
            elif name == "</s>":
                eos = idx
            elif name == "<pad>":
                pad = idx
    charsmap = norm.get(2, b"")
    flags = 0
    if norm.get(3, 1):
        flags |= 1  # add_dummy_prefix
    if norm.get(4, 1):
        flags |= 2  # remove_extra_whitespaces
    if norm.get(5, 1):
        flags |= 4  # escape_whitespaces
    if trainer.get(35, 0):
        flags |= 8  # byte_fallback

    trie_size = struct.unpack("<I", charsmap[:4])[0] if len(charsmap) > 4 else 0
    trie = charsmap[4 : 4 + trie_size]
    blob = charsmap[4 + trie_size :]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<IIIII", len(pieces), unk, bos, eos, pad))
        fh.write(struct.pack("<I", flags))
        fh.write(struct.pack("<I", len(trie)))
        fh.write(trie)
        fh.write(struct.pack("<I", len(blob)))
        fh.write(blob)
        for piece, score, ptype in pieces:
            fh.write(struct.pack("<BfH", ptype, score, len(piece)))
            fh.write(piece)

    summary = {
        "out": str(out),
        "kb": round(out.stat().st_size / 1024, 1),
        "pieces": len(pieces),
        "unk": unk,
        "bos": bos,
        "eos": eos,
        "pad": pad,
        "flags": flags,
        "trie_units": len(trie) // 4,
        "blob": len(blob),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
