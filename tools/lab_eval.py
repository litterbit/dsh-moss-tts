"""对一份权重在固定语料上评精度：teacher forcing + 与 f16 参考的 logits 对比。

用法：
    python tools/lab_eval.py --weights weights/llm_t2.moss [--corpora zh_mid,zh_long,en,clone]

输出：每句的 flip / KL / top-25，以及合计。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "csrc" / "build" / "moss_llm.exe"
LAB = ROOT / "runs" / "lab"


def softmax(x: np.ndarray, t: float) -> np.ndarray:
    z = x.astype(np.float64) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 1024)


def compare(ref: Path, test: Path, temperature: float = 0.8, k: int = 25) -> dict:
    a, b = load(ref), load(test)
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    p, q = softmax(a, temperature), softmax(b, temperature)
    kl = np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=1)
    top1 = (a.argmax(axis=1) == b.argmax(axis=1)).astype(np.float64)
    ia = np.argpartition(-a, k, axis=1)[:, :k]
    ib = np.argpartition(-b, k, axis=1)[:, :k]
    overlap = np.array([len(set(ia[i].tolist()) & set(ib[i].tolist())) / k for i in range(n)])
    return {"steps": int(n), "kl": float(kl.mean()), "top1": float(top1.mean()),
            "top25": float(overlap.mean())}


def evaluate(weights: Path, corpora: list[str]) -> dict:
    out_rows = {}
    for name in corpora:
        d = LAB / name
        prompt, codes = d / "prompt.txt", d / "codes.txt"
        frames = len([ln for ln in codes.read_text(encoding="utf-8").splitlines() if ln.strip()])
        dump = d / f"eval_{weights.stem}.bin"
        proc = subprocess.run([str(EXE), "--weights", str(weights), "--prompt", str(prompt),
                               "--frames", str(frames), "--threads", "4", "--greedy", "--force-codes", str(codes),
                               "--dump-logits", str(dump)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        flip = re.search(r'"teacher_flip_rate":\s*([0-9.]+)', proc.stdout or "")
        cmp = compare(d / "ref_f16.bin", dump)
        cmp["flip"] = float(flip.group(1)) if flip else None
        out_rows[name] = cmp
        dump.unlink(missing_ok=True)
    return out_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--corpora", default="zh_mid,zh_long,en,clone")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    weights = Path(args.weights)
    corpora = [c for c in args.corpora.split(",") if c]
    rows = evaluate(weights, corpora)
    total = {"kl": float(np.mean([r["kl"] for r in rows.values()])),
             "flip": float(np.mean([r["flip"] for r in rows.values() if r["flip"] is not None])),
             "top1": float(np.mean([r["top1"] for r in rows.values()])),
             "top25": float(np.mean([r["top25"] for r in rows.values()]))}
    print(f"{'corpus':8s}{'flip':>8s}{'KL':>10s}{'top1':>8s}{'top25':>8s}")
    for name, r in rows.items():
        print(f"{name:8s}{r['flip']:8.3f}{r['kl']:10.5f}{r['top1']:8.3f}{r['top25']:8.3f}")
    print(f"{'MEAN':8s}{total['flip']:8.3f}{total['kl']:10.5f}{total['top1']:8.3f}{total['top25']:8.3f}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"weights": str(weights), "rows": rows, "mean": total},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
