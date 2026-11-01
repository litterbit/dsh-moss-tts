"""逐张量 Q4 消融：每个候选组单独降到 Q4_0_4x4，其余保持 Q8_0_4x4。

对每个方案：导出权重 → C++ teacher-forced 跑一遍 → 统计 ms/帧、code 翻转、KL、top-25 重叠。
用法：
    python tools/sweep_q4.py --prompt runs/cpp/prompt.txt --codes runs/cpp/codes_f16_24.txt \
        --ref-logits runs/ablation/llm_f16.bin --out runs/ablation
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
PY = sys.executable
EXE = ROOT / "csrc" / "build" / "moss_llm.exe"
CKPT = Path(r"D:\workplace\audio\github\MOSS-TTS-Nano-100M\pytorch_model.bin")

BASE = {"default": "q8_0_4x4", "overrides": []}

GROUPS: dict[str, list[str]] = {
    "base_q8": [],
    "g_attn": ["g*.attn.w"],
    "g_proj": ["g*.proj.w"],
    "g_fc_in": ["g*.fc_in.w"],
    "g_fc_out": ["g*.fc_out.w"],
    "local_all": ["local.*.w"],
    "heads": ["aemb.*", "wte"],
    "g_all_but_fc_in": ["g*.attn.w", "g*.proj.w", "g*.fc_out.w"],
}


def export(plan: dict, out_path: Path) -> None:
    plan_file = out_path.with_suffix(".plan.json")
    plan_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(
        [PY, str(ROOT / "tools" / "export_weights.py"), "--ckpt", str(CKPT),
         "--out", str(out_path), "--plan-file", str(plan_file)],
        check=True, capture_output=True,
    )


def run_engine(weights: Path, prompt: Path, codes: Path, frames: int, logits: Path | None) -> tuple[dict, list[list[int]]]:
    cmd = [str(EXE), "--weights", str(weights), "--prompt", str(prompt), "--frames", str(frames),
           "--greedy", "--threads", "4", "--force-codes", str(codes)]
    if logits is not None:
        cmd += ["--dump-logits", str(logits)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"engine failed: {proc.stderr[:400]}")
    metrics = {}
    for key in ("prefill_ms", "total_ms_per_frame", "teacher_flip_rate"):
        m = re.search(rf'"{key}": ([0-9.]+)', proc.stdout)
        if m:
            metrics[key] = float(m.group(1))
    start = proc.stdout.find("\"codes\": [")
    codes_out: list[list[int]] = []
    if start >= 0:
        block = proc.stdout[start:].split("]")[:-1]
        for line in block[1:]:
            nums = re.findall(r"\d+", line)
            if len(nums) == 16:
                codes_out.append([int(x) for x in nums])
    return metrics, codes_out


def load_logits(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    return raw.reshape(-1, 1024)


def softmax(x: np.ndarray, t: float) -> np.ndarray:
    z = x.astype(np.float64) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compare(ref: np.ndarray, test: np.ndarray, topk: int, temperature: float) -> dict:
    n = min(ref.shape[0], test.shape[0])
    ref, test = ref[:n], test[:n]
    p, q = softmax(ref, temperature), softmax(test, temperature)
    kl = float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=1)))
    ia = np.argpartition(-ref, topk, axis=1)[:, :topk]
    ib = np.argpartition(-test, topk, axis=1)[:, :topk]
    overlap = float(np.mean([len(set(ia[i]) & set(ib[i])) / topk for i in range(n)]))
    top1 = float(np.mean(ref.argmax(axis=1) == test.argmax(axis=1)))
    return {"kl": round(kl, 5), "topk_overlap": round(overlap, 4), "top1": round(top1, 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--ref-logits", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=1.7)
    parser.add_argument("--groups", default=None, help="逗号分隔，默认全部")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = load_logits(Path(args.ref_logits))
    names = args.groups.split(",") if args.groups else list(GROUPS)

    rows = []
    for name in names:
        patterns = GROUPS[name]
        plan = {"default": BASE["default"],
                "overrides": [{"pattern": p, "type": "q4_0_4x4"} for p in patterns]}
        weights = out / f"plan_{name}.moss"
        export(plan, weights)
        logits_path = out / f"plan_{name}.logits.bin"
        metrics, codes = run_engine(weights, Path(args.prompt), Path(args.codes), args.frames, logits_path)
        cmp = compare(ref, load_logits(logits_path), args.topk, args.temperature)
        rows.append({"plan": name, **metrics, **cmp})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    (out / "sweep_result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n| plan | ms/帧 | flip | KL | top25 |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['plan']} | {r.get('total_ms_per_frame', 0):.2f} | "
              f"{r.get('teacher_flip_rate', 0):.3f} | {r['kl']:.5f} | {r['topk_overlap']:.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
