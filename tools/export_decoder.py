"""把 MOSS-Audio-Tokenizer-Nano 的解码器权重导出成 .moss。

只导出解码路径需要的东西：
  quantizer: input_proj / output_proj / 16 路 LFQ（codebook + out_proj）
  decoder.{1,3,5,7}: input_proj / output_proj / transformer 各层

weight-norm（`parametrizations.weight.original0/original1`）在这里合并成普通矩阵：
    w = g * v / ||v||，其中 ||v|| 沿 dim 1..N 归一化（torch.nn.utils.weight_norm 语义）

用法：
    python tools/export_decoder.py --src <safetensors> --out weights/decoder_f32.moss --dtype f32
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from export_weights import DTYPE_F16, DTYPE_F32, DTYPE_Q8_0, write_moss  # noqa: E402


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """不依赖 safetensors 库：直接解析头 + mmap 数据区。"""
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        base = 8 + n
        out: dict[str, np.ndarray] = {}
        with path.open("rb") as data_fh:
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                dtype = info["dtype"]
                if dtype != "F32":
                    raise ValueError(f"{name}: 只支持 F32，实际 {dtype}")
                shape = tuple(info["shape"])
                start, end = info["data_offsets"]
                data_fh.seek(base + start)
                raw = data_fh.read(end - start)
                out[name] = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        return out


def merge_weight_norm(tensors: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """parametrizations.weight.original0/1 -> 普通权重。"""
    g = tensors[prefix + ".parametrizations.weight.original0"]   # [out, 1, 1]
    v = tensors[prefix + ".parametrizations.weight.original1"]   # [out, in, 1]
    norm = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1).reshape(-1, 1, 1)
    return (g * v / np.maximum(norm, 1e-12)).reshape(v.shape[0], v.shape[1])


def quantize_rows(w: np.ndarray) -> tuple[np.ndarray, float]:
    rows = w.reshape(w.shape[0], -1) if w.ndim > 2 else w
    amax = float(np.abs(rows).max())
    return rows.astype(np.float32), amax


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="f32", choices=["f32", "f16", "q8"])
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    st = load_safetensors(Path(args.src))
    dtype = {"f32": DTYPE_F32, "f16": DTYPE_F16, "q8": DTYPE_Q8_0}[args.dtype]

    entries: list[tuple[str, int, tuple[int, int], bytes]] = []
    report: list[dict] = []

    def add(name: str, w: np.ndarray, *, quantizable: bool = True) -> None:
        arr = np.asarray(w, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        assert arr.ndim == 2, (name, arr.shape)
        rows, cols = arr.shape
        use = dtype if (quantizable and dtype != DTYPE_F32 and rows >= 1 and cols % 32 == 0) else DTYPE_F32
        if use == DTYPE_Q8_0:
            from export_weights import quantize_q8_0

            blob = quantize_q8_0(arr).tobytes()
        elif use == DTYPE_F16:
            blob = arr.astype(np.float16).tobytes()
        else:
            blob = arr.tobytes()
        entries.append((name, use, (rows, cols), blob))
        report.append({"name": name, "dtype": use, "shape": [rows, cols], "bytes": len(blob)})

    # quantizer
    add("q.in.w", merge_weight_norm(st, "quantizer.input_proj"))
    add("q.in.b", st["quantizer.input_proj.bias"], quantizable=False)
    add("q.out.w", merge_weight_norm(st, "quantizer.output_proj"))
    add("q.out.b", st["quantizer.output_proj.bias"], quantizable=False)
    for i in range(16):
        p = f"quantizer.quantizers.{i}"
        add(f"q{i}.cb", st[f"{p}.codebook.weight"])
        add(f"q{i}.out.w", merge_weight_norm(st, f"{p}.out_proj"))
        add(f"q{i}.out.b", st[f"{p}.out_proj.bias"], quantizable=False)

    # decoder stages
    for stage in (1, 3, 5, 7):
        p = f"decoder.{stage}"
        add(f"d{stage}.in.w", st[f"{p}.input_proj.weight"])
        add(f"d{stage}.out.w", st[f"{p}.output_proj.weight"])
        layers = 0
        while f"{p}.transformer.layers.{layers}.norm1.weight" in st:
            layers += 1
        for n in range(layers):
            lp = f"{p}.transformer.layers.{n}"
            add(f"d{stage}.l{n}.n1.w", st[f"{lp}.norm1.weight"], quantizable=False)
            add(f"d{stage}.l{n}.n1.b", st[f"{lp}.norm1.bias"], quantizable=False)
            add(f"d{stage}.l{n}.n2.w", st[f"{lp}.norm2.weight"], quantizable=False)
            add(f"d{stage}.l{n}.n2.b", st[f"{lp}.norm2.bias"], quantizable=False)
            add(f"d{stage}.l{n}.attn.w", st[f"{lp}.self_attn.in_proj.weight"])
            add(f"d{stage}.l{n}.proj.w", st[f"{lp}.self_attn.out_proj.weight"])
            add(f"d{stage}.l{n}.f1.w", st[f"{lp}.ffn.0.weight"])
            add(f"d{stage}.l{n}.f2.w", st[f"{lp}.ffn.2.weight"])
            add(f"d{stage}.l{n}.ls1", st[f"{lp}.layer_scale_1.scale"], quantizable=False)
            add(f"d{stage}.l{n}.ls2", st[f"{lp}.layer_scale_2.scale"], quantizable=False)
        report.append({"stage": stage, "layers": layers})

    out = Path(args.out)
    write_moss(out, entries)
    summary = {"src": str(args.src), "dtype": args.dtype, "out": str(out),
               "bytes": out.stat().st_size, "tensors": len(entries), "stages": report[-4:]}
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "mb": round(out.stat().st_size / 1048576, 2),
                      "tensors": len(entries)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
