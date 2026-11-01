"""把 MOSS-Audio-Tokenizer-Nano 的编码器权重导出成 .moss（克隆用）。

编码路径需要：
  quantizer: input_proj / 16 路 LFQ（codebook + in_proj + out_proj）
  encoder.{1,3,5,7}: input_proj / output_proj / transformer 各层

weight-norm（`parametrizations.weight.original0/original1`）在这里合并成普通矩阵：
    w = g * v / ||v||，沿 dim 1..N 归一化（torch.nn.utils.weight_norm 语义）

用法：
    python tools/export_encoder.py --src <safetensors> --out weights/encoder_f16.moss --dtype f16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from export_decoder import load_safetensors, merge_weight_norm  # noqa: E402
from export_weights import DTYPE_F16, DTYPE_F32, DTYPE_Q8_0, write_moss  # noqa: E402

STAGES = (1, 3, 5, 7)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="f16", choices=["f32", "f16", "q8"])
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

    # ---- quantizer（编码方向：z -> 512 -> 16×LFQ）----
    add("q.in.w", merge_weight_norm(st, "quantizer.input_proj"))
    add("q.in.b", st["quantizer.input_proj.bias"], quantizable=False)
    for i in range(16):
        p = f"quantizer.quantizers.{i}"
        add(f"q{i}.cb", st[f"{p}.codebook.weight"], quantizable=False)
        add(f"q{i}.in.w", merge_weight_norm(st, f"{p}.in_proj"))
        add(f"q{i}.in.b", st[f"{p}.in_proj.bias"], quantizable=False)
        add(f"q{i}.out.w", merge_weight_norm(st, f"{p}.out_proj"))
        add(f"q{i}.out.b", st[f"{p}.out_proj.bias"], quantizable=False)

    # ---- encoder stages ----
    for stage in STAGES:
        p = f"encoder.{stage}"
        add(f"e{stage}.in.w", st[f"{p}.input_proj.weight"])
        add(f"e{stage}.out.w", st[f"{p}.output_proj.weight"])
        layers = 0
        while f"{p}.transformer.layers.{layers}.norm1.weight" in st:
            layers += 1
        for n in range(layers):
            lp = f"{p}.transformer.layers.{n}"
            add(f"e{stage}.l{n}.n1.w", st[f"{lp}.norm1.weight"], quantizable=False)
            add(f"e{stage}.l{n}.n1.b", st[f"{lp}.norm1.bias"], quantizable=False)
            add(f"e{stage}.l{n}.n2.w", st[f"{lp}.norm2.weight"], quantizable=False)
            add(f"e{stage}.l{n}.n2.b", st[f"{lp}.norm2.bias"], quantizable=False)
            add(f"e{stage}.l{n}.attn.w", st[f"{lp}.self_attn.in_proj.weight"])
            add(f"e{stage}.l{n}.proj.w", st[f"{lp}.self_attn.out_proj.weight"])
            add(f"e{stage}.l{n}.f1.w", st[f"{lp}.ffn.0.weight"])
            add(f"e{stage}.l{n}.f2.w", st[f"{lp}.ffn.2.weight"])
            add(f"e{stage}.l{n}.ls1", st[f"{lp}.layer_scale_1.scale"], quantizable=False)
            add(f"e{stage}.l{n}.ls2", st[f"{lp}.layer_scale_2.scale"], quantizable=False)
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
