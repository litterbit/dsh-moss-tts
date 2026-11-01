"""把 PyTorch checkpoint 导出成 .moss 权重文件（支持 fp16 / Q8_0 / Q4_0 混合）。

用法：
    python tools/export_weights.py --ckpt <pytorch_model.bin> --out weights/llm_q8.moss --plan q8
    python tools/export_weights.py --ckpt <...> --out weights/llm_f16.moss --plan f16

.plan 内建三套：f16 / q8 / q4；也可以用 --plan-file 指定 JSON（阶段二消融用）。

权重文件布局见 doc/推理加速计划.md 5.1。块格式对齐 GGML：
    Q8_0: [fp16 d][int8 q x32]              = 34 B / 32 权重
    Q4_0: [fp16 d][uint8 qs x16]            = 18 B / 32 权重
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MAGIC = b"MOSW"
VERSION = 1
HEADER_SIZE = 16
ENTRY_SIZE = 80
NAME_SIZE = 48
DATA_ALIGN = 32

DTYPE_F16, DTYPE_Q8_0, DTYPE_Q4_0, DTYPE_F32 = 0, 1, 2, 3
DTYPE_NAMES = {DTYPE_F16: "f16", DTYPE_Q8_0: "q8_0", DTYPE_Q4_0: "q4_0", DTYPE_F32: "f32",
               4: "q8_0_4x4", 5: "q4_0_4x4"}

# 每帧被读的次数（用于导出报告，不是推理参数）
GLOBAL_LAYERS = 12


@dataclass(frozen=True)
class TensorSpec:
    name: str        # .moss 里的名字
    source: str      # checkpoint 里的名字
    kind: str        # "matrix" | "vector" | "embed" —— 决定能否量化
    rows: int
    cols: int


def build_specs() -> list[TensorSpec]:
    """列出 LLM 段需要的全部张量（head 与 embedding 共享，只存一份）。"""
    specs: list[TensorSpec] = [
        TensorSpec("wte", "transformer.wte.weight", "embed", 16384, 768),
        TensorSpec("lnf.w", "transformer.ln_f.weight", "vector", 1, 768),
        TensorSpec("lnf.b", "transformer.ln_f.bias", "vector", 1, 768),
    ]
    for i in range(GLOBAL_LAYERS):
        prefix = f"transformer.h.{i}"
        specs += [
            TensorSpec(f"g{i}.ln1.w", f"{prefix}.ln_1.weight", "vector", 1, 768),
            TensorSpec(f"g{i}.ln1.b", f"{prefix}.ln_1.bias", "vector", 1, 768),
            TensorSpec(f"g{i}.attn.w", f"{prefix}.attn.c_attn.weight", "matrix", 2304, 768),
            TensorSpec(f"g{i}.attn.b", f"{prefix}.attn.c_attn.bias", "vector", 1, 2304),
            TensorSpec(f"g{i}.proj.w", f"{prefix}.attn.c_proj.weight", "matrix", 768, 768),
            TensorSpec(f"g{i}.proj.b", f"{prefix}.attn.c_proj.bias", "vector", 1, 768),
            TensorSpec(f"g{i}.ln2.w", f"{prefix}.ln_2.weight", "vector", 1, 768),
            TensorSpec(f"g{i}.ln2.b", f"{prefix}.ln_2.bias", "vector", 1, 768),
            TensorSpec(f"g{i}.fc_in.w", f"{prefix}.mlp.fc_in.weight", "matrix", 3072, 768),
            TensorSpec(f"g{i}.fc_in.b", f"{prefix}.mlp.fc_in.bias", "vector", 1, 3072),
            TensorSpec(f"g{i}.fc_out.w", f"{prefix}.mlp.fc_out.weight", "matrix", 768, 3072),
            TensorSpec(f"g{i}.fc_out.b", f"{prefix}.mlp.fc_out.bias", "vector", 1, 768),
        ]
    for ch in range(16):
        specs.append(TensorSpec(f"aemb.{ch}", f"audio_embeddings.{ch}.weight", "embed", 1024, 768))
    specs += [
        TensorSpec("local.ln1.w", "local_transformer.h.0.ln_1.weight", "vector", 1, 768),
        TensorSpec("local.ln1.b", "local_transformer.h.0.ln_1.bias", "vector", 1, 768),
        TensorSpec("local.attn.w", "local_transformer.h.0.attn.c_attn.weight", "matrix", 2304, 768),
        TensorSpec("local.attn.b", "local_transformer.h.0.attn.c_attn.bias", "vector", 1, 2304),
        TensorSpec("local.proj.w", "local_transformer.h.0.attn.c_proj.weight", "matrix", 768, 768),
        TensorSpec("local.proj.b", "local_transformer.h.0.attn.c_proj.bias", "vector", 1, 768),
        TensorSpec("local.ln2.w", "local_transformer.h.0.ln_2.weight", "vector", 1, 768),
        TensorSpec("local.ln2.b", "local_transformer.h.0.ln_2.bias", "vector", 1, 768),
        TensorSpec("local.fc_in.w", "local_transformer.h.0.mlp.fc_in.weight", "matrix", 3072, 768),
        TensorSpec("local.fc_in.b", "local_transformer.h.0.mlp.fc_in.bias", "vector", 1, 3072),
        TensorSpec("local.fc_out.w", "local_transformer.h.0.mlp.fc_out.weight", "matrix", 768, 3072),
        TensorSpec("local.fc_out.b", "local_transformer.h.0.mlp.fc_out.bias", "vector", 1, 768),
        TensorSpec("local.lnf.w", "local_transformer.ln_f.weight", "vector", 1, 768),
        TensorSpec("local.lnf.b", "local_transformer.ln_f.bias", "vector", 1, 768),
    ]
    return specs


# ---------------------------------------------------------------- 量化

def quantize_q8_0(w: np.ndarray) -> np.ndarray:
    """w: [rows, K] float32，K 必须是 32 的倍数。返回 [rows, K/32*34] uint8。"""
    rows, k = w.shape
    assert k % 32 == 0, k
    blocks = w.reshape(rows, k // 32, 32)
    amax = np.abs(blocks).max(axis=2, keepdims=True)
    d = (amax / 127.0).astype(np.float16)
    id_ = np.where(d.astype(np.float32) == 0, 0.0, 1.0 / d.astype(np.float32))
    q = np.clip(np.rint(blocks * id_), -127, 127).astype(np.int8)

    out = np.empty((rows, k // 32, 34), dtype=np.uint8)
    out[:, :, 0:2] = d.view(np.uint8).reshape(rows, k // 32, 2)
    out[:, :, 2:] = q.view(np.uint8)
    return out.reshape(rows, -1)


def quantize_q4_0(w: np.ndarray) -> np.ndarray:
    """w: [rows, K] float32。返回 [rows, K/32*18] uint8。"""
    rows, k = w.shape
    assert k % 32 == 0, k
    blocks = w.reshape(rows, k // 32, 32)
    flat = blocks.reshape(-1, 32)
    idx = np.abs(flat).argmax(axis=1)
    max_signed = flat[np.arange(flat.shape[0]), idx]
    d = (max_signed / -8.0).astype(np.float16)
    id_ = np.where(d.astype(np.float32) == 0, 0.0, 1.0 / d.astype(np.float32)).reshape(-1, 1)
    q = np.clip(np.rint(flat * id_ + 8.5), 0, 15).astype(np.uint8).reshape(rows, k // 32, 32)

    lo = q[:, :, 0::2] & 0x0F
    hi = (q[:, :, 1::2] & 0x0F) << 4
    packed = (lo | hi).astype(np.uint8)

    out = np.empty((rows, k // 32, 18), dtype=np.uint8)
    out[:, :, 0:2] = d.view(np.uint8).reshape(rows, k // 32, 2)
    out[:, :, 2:] = packed
    return out.reshape(rows, -1)


def encode(tensor: torch.Tensor, dtype: int) -> tuple[bytes, tuple[int, int]]:
    w = tensor.detach().to(torch.float32).numpy()
    if w.ndim == 1:
        w = w.reshape(1, -1)
    if dtype == DTYPE_F32:
        return w.astype(np.float32).tobytes(), w.shape
    if dtype == DTYPE_F16:
        return w.astype(np.float16).tobytes(), w.shape
    if dtype == DTYPE_Q8_0:
        return quantize_q8_0(w).tobytes(), w.shape
    if dtype == DTYPE_Q4_0:
        return quantize_q4_0(w).tobytes(), w.shape
    if dtype == DTYPE_Q8_0_4X4:
        return pack_q8_4x4(w).tobytes(), w.shape
    if dtype == DTYPE_Q4_0_4X4:
        return pack_q4_4x4(w).tobytes(), w.shape
    raise ValueError(dtype)



# ---------------------------------------------------------------- 4x4 repack

DTYPE_Q8_0_4X4, DTYPE_Q4_0_4X4 = 4, 5


def pack_q8_4x4(w: np.ndarray) -> np.ndarray:
    """[rows, K] -> [groups, K/32, 136]：每组 4 行，块内 4 个 scale 相邻。"""
    rows, k = w.shape
    if rows % 4 != 0:
        raise ValueError(f"rows={rows} 不是 4 的倍数")
    nb = k // 32
    blocks = w.reshape(rows, nb, 32)
    amax = np.abs(blocks).max(axis=2, keepdims=True)
    d = (amax / 127.0).astype(np.float16)
    id_ = np.where(d.astype(np.float32) == 0, 0.0, 1.0 / d.astype(np.float32))
    q = np.clip(np.rint(blocks * id_), -127, 127).astype(np.int8)

    groups = rows // 4
    d2 = np.ascontiguousarray(d.reshape(rows, nb)).view(np.uint8)   # [rows, nb, 2]
    d_g = d2.reshape(groups, 4, nb, 2).transpose(0, 2, 1, 3)        # [g, nb, 4, 2]
    q_g = np.ascontiguousarray(q.reshape(groups, 4, nb, 32).transpose(0, 2, 1, 3)).view(np.uint8)
    out = np.empty((groups, nb, 8 + 128), dtype=np.uint8)
    out[:, :, 0:8] = d_g.reshape(groups, nb, 8)
    out[:, :, 8:] = q_g.reshape(groups, nb, 128)
    return out.reshape(groups, -1)


def pack_q4_4x4(w: np.ndarray) -> np.ndarray:
    """[rows, K] -> [groups, K/32, 72]：4 个 scale + 4×16 nibble。"""
    rows, k = w.shape
    if rows % 4 != 0:
        raise ValueError(f"rows={rows} 不是 4 的倍数")
    nb = k // 32
    blocks = w.reshape(rows, nb, 32)
    flat = blocks.reshape(-1, 32)
    idx = np.abs(flat).argmax(axis=1)
    max_signed = flat[np.arange(flat.shape[0]), idx]
    d = (max_signed / -8.0).astype(np.float16)
    id_ = np.where(d.astype(np.float32) == 0, 0.0, 1.0 / d.astype(np.float32)).reshape(-1, 1)
    q = np.clip(np.rint(flat * id_ + 8.5), 0, 15).astype(np.uint8).reshape(rows, nb, 32)
    lo = q[:, :, 0::2] & 0x0F
    hi = (q[:, :, 1::2] & 0x0F) << 4
    packed = (lo | hi).astype(np.uint8)                            # [rows, nb, 16]

    groups = rows // 4
    d2 = np.ascontiguousarray(d.reshape(rows, nb)).view(np.uint8)   # [rows, nb, 2]
    d_g = d2.reshape(groups, 4, nb, 2).transpose(0, 2, 1, 3)        # [g, nb, 4, 2]
    p_g = np.ascontiguousarray(packed.reshape(groups, 4, nb, 16).transpose(0, 2, 1, 3))
    out = np.empty((groups, nb, 8 + 64), dtype=np.uint8)
    out[:, :, 0:8] = d_g.reshape(groups, nb, 8)
    out[:, :, 8:] = p_g.reshape(groups, nb, 64)
    return out.reshape(groups, -1)
# ---------------------------------------------------------------- 量化方案

BUILTIN_PLANS: dict[str, dict] = {
    "f16": {"default": "f16", "overrides": []},
    "q8": {"default": "q8_0", "overrides": [{"pattern": "*", "type": "q8_0"}]},
    "q4": {"default": "q4_0", "overrides": []},
    "q8x4": {"default": "q8_0_4x4", "overrides": []},
    "q4x4": {"default": "q4_0_4x4", "overrides": []},
}


def resolve_dtype(spec: TensorSpec, plan: dict) -> int:
    # 向量（LN/bias）永远 fp32：体积可忽略，但对数值敏感
    if spec.kind == "vector":
        return DTYPE_F32
    dtype_name = plan.get("default", "q8_0")
    for rule in plan.get("overrides", []):
        if fnmatch.fnmatchcase(spec.name, rule["pattern"]):
            dtype_name = rule["type"]
            break
    if dtype_name == "f16":
        return DTYPE_F16
    if dtype_name == "q8_0":
        return DTYPE_Q8_0
    if dtype_name == "q4_0":
        return DTYPE_Q4_0
    if dtype_name == "q8_0_4x4":
        return DTYPE_Q8_0_4X4
    if dtype_name == "q4_0_4x4":
        return DTYPE_Q4_0_4X4
    if dtype_name == "f32":
        return DTYPE_F32
    raise ValueError(f"unknown dtype {dtype_name}")


# ---------------------------------------------------------------- 写文件

def write_moss(path: Path, entries: list[tuple[str, int, tuple[int, int], bytes]]) -> None:
    """entries: (name, dtype, (rows, cols), payload)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = MAGIC + struct.pack("<III", VERSION, len(entries), 0)

    table = bytearray()
    payload = bytearray()
    data_start = HEADER_SIZE + ENTRY_SIZE * len(entries)
    for name, dtype, (rows, cols), blob in entries:
        offset = data_start + len(payload)
        pad = (-offset) % DATA_ALIGN
        payload += b"\x00" * pad
        offset = data_start + len(payload)
        name_bytes = name.encode("ascii")
        assert len(name_bytes) < NAME_SIZE, name
        table += name_bytes.ljust(NAME_SIZE, b"\x00")
        table += struct.pack("<BBH", dtype, 2, 0)
        table += struct.pack("<III", rows, cols, 1)
        table += struct.pack("<QQ", offset, len(blob))
        payload += blob

    with path.open("wb") as fh:
        fh.write(header)
        fh.write(table)
        fh.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--plan", default="q8", choices=sorted(BUILTIN_PLANS))
    parser.add_argument("--plan-file", default=None)
    parser.add_argument("--report", default=None, help="额外输出张量清单 JSON")
    args = parser.parse_args(argv)

    plan = json.loads(Path(args.plan_file).read_text(encoding="utf-8")) if args.plan_file else BUILTIN_PLANS[args.plan]
    ckpt = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=True)

    entries: list[tuple[str, int, tuple[int, int], bytes]] = []
    report: list[dict] = []
    for spec in build_specs():
        if spec.source not in ckpt:
            raise KeyError(f"checkpoint 缺少张量: {spec.source}")
        tensor = ckpt[spec.source]
        dtype = resolve_dtype(spec, plan)
        blob, shape = encode(tensor, dtype)
        if tuple(shape) != (spec.rows, spec.cols):
            raise ValueError(f"{spec.name} 形状不符: {shape} != {(spec.rows, spec.cols)}")
        entries.append((spec.name, dtype, shape, blob))
        report.append({"name": spec.name, "dtype": DTYPE_NAMES[dtype],
                       "shape": list(shape), "bytes": len(blob)})

    out = Path(args.out)
    write_moss(out, entries)
    total = sum(item["bytes"] for item in report)

    summary = {"source": str(args.ckpt), "plan": plan, "file": str(out),
               "file_bytes": out.stat().st_size, "payload_bytes": total,
               "tensors": report}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"file": str(out), "bytes": out.stat().st_size,
                      "mb": round(out.stat().st_size / 1048576, 1),
                      "tensors": len(entries)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
