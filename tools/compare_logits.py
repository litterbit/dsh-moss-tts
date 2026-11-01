"""比较两份 audio logits dump（C++ 引擎 --dump-logits 的原始 float32 输出）。

用法：
    python tools/compare_logits.py --ref ref.bin --test q8.bin [--topk 25]

输出：KL(ref||test)、top-k 重叠率、top-1 一致率、每帧明细。
dump 的布局：frames × 16 通道 × 1024 个 logits，float32 连续存放。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_CODEBOOK = 1024


def load(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % N_CODEBOOK != 0:
        raise SystemExit(f"{path}: 大小 {raw.size} 不是 {N_CODEBOOK} 的整数倍")
    return raw.reshape(-1, N_CODEBOOK)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x.astype(np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=1)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    ia = np.argpartition(-a, k, axis=1)[:, :k]
    ib = np.argpartition(-b, k, axis=1)[:, :k]
    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        out[i] = len(set(ia[i].tolist()) & set(ib[i].tolist())) / k
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=1.7, help="采样温度（与推理一致才可比）")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    ref = load(args.ref)
    test = load(args.test)
    steps = min(ref.shape[0], test.shape[0])
    ref, test = ref[:steps], test[:steps]

    p = softmax(ref, args.temperature)
    q = softmax(test, args.temperature)
    kl = kl_divergence(p, q)
    overlap = topk_overlap(ref, test, args.topk)
    top1 = (ref.argmax(axis=1) == test.argmax(axis=1)).astype(np.float64)

    per_frame = []
    frames = steps // 16
    for f in range(frames):
        sl = slice(f * 16, (f + 1) * 16)
        per_frame.append(
            {"frame": f, "kl": round(float(kl[sl].mean()), 6),
             "topk_overlap": round(float(overlap[sl].mean()), 4),
             "top1": round(float(top1[sl].mean()), 4)}
        )

    summary = {
        "steps": steps,
        "frames": frames,
        "kl_mean": round(float(kl.mean()), 6),
        "kl_p95": round(float(np.percentile(kl, 95)), 6),
        "topk_overlap_mean": round(float(overlap.mean()), 4),
        "top1_agreement": round(float(top1.mean()), 4),
        "per_frame": per_frame,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_frame"}, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
