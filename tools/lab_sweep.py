"""逐张量组的精度敏感度扫描（QLM 优化第一步）。

回答：把某一组权重降到 Q4，对「prefill」和「decode」分别造成多大精度损失，
换到多少每帧字节。用现成的 --save-state / --load-state 把两段拆开。

用法：
    python tools/lab_sweep.py setup                         # 建语料 + f16 codes + 参考 dump
    python tools/lab_sweep.py sweep --corpora zh_mid,zh_long
    python tools/lab_sweep.py sweep --groups L,F0,G0
    python tools/lab_sweep.py report

产物：runs/lab/{corpus}/prompt.txt、codes.txt、ref_f16.bin、ref_q8.bin、state_{f16,q8}.bin
      runs/lab/plans/*.json、runs/lab/w/*.moss、runs/lab/results.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
EXE = ROOT / "csrc" / "build" / "moss_llm.exe"
CKPT = Path(r"D:\workplace\audio\github\MOSS-TTS-Nano-100M\pytorch_model.bin")
F16 = ROOT / "weights" / "llm_f16.moss"
Q8 = ROOT / "weights" / "llm_q8x4.moss"
LAB = ROOT / "runs" / "lab"

CORPORA = {
    "zh_mid": {
        "text": "今天天气不错，我们一起出门走走吧。",
        "voice": "Junhao",
        "frames": 60,
    },
    "zh_long": {
        "text": "人工智能正在改变我们的生活方式。从语音助手到自动驾驶，从医疗诊断到金融风控，"
                "这项技术已经渗透到各个行业。但与此同时，如何保证系统的可靠性和公平性，也成为人们关注的焦点。",
        "voice": "Junhao",
        "frames": 150,
    },
    "en": {
        "text": "The quick brown fox jumps over the lazy dog, and the weather is lovely today.",
        "voice": "Ava",
        "frames": 50,
    },
    "clone": {
        "text": "今天天气不错，我们一起出门走走吧。",
        "voice": "deepseek_letmogo",
        "frames": 60,
        "profile": ROOT / "web" / "voices" / "deepseek_letmogo.mossclone",
    },
}

Q8_BYTES_PER_WEIGHT = 34 / 32  # q8_0：32 权重 34 字节
Q4_BYTES_PER_WEIGHT = 18 / 32  # q4_0：32 权重 18 字节


def group_specs() -> list[dict]:
    """每组：名字、匹配模式、权重个数、每帧重读次数。"""
    groups: list[dict] = []
    for i in range(12):
        groups.append({"name": f"G{i}", "patterns": [f"g{i}.attn.w"],
                       "weights": 2304 * 768, "per_frame": 1, "label": f"layer{i} attn"})
    for i in range(12):
        groups.append({"name": f"P{i}", "patterns": [f"g{i}.proj.w"],
                       "weights": 768 * 768, "per_frame": 1, "label": f"layer{i} proj"})
    for i in range(12):
        groups.append({"name": f"F{i}", "patterns": [f"g{i}.fc_in.w", f"g{i}.fc_out.w"],
                       "weights": 2 * 3072 * 768, "per_frame": 1, "label": f"layer{i} fc"})
    groups.append({"name": "L", "patterns": ["local.*"],
                   "weights": 4 * 2304 * 768 + 2 * 768 * 768 + 2 * 3072 * 768,
                   "per_frame": 17, "label": "local x17"})
    groups.append({"name": "E-aemb", "patterns": ["aemb.*"],
                   "weights": 16 * 1024 * 768, "per_frame": 1, "label": "16 codebook heads"})
    groups.append({"name": "E-wte", "patterns": ["wte"],
                   "weights": 16384 * 768, "per_frame": 1, "label": "vocab head"})
    return groups


def run_json(args: list[str], timeout: int = 1200) -> dict:
    proc = subprocess.run([str(a) for a in args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    out = proc.stdout or ""
    start = out.find("{")
    if start < 0:
        raise RuntimeError(f"{args[0]} 无 JSON 输出\nstdout={out[:400]}\nstderr={proc.stderr[:400]}")
    return json.loads(out[start:])


def run_quiet(args: list[str], timeout: int = 1200) -> None:
    proc = subprocess.run([str(a) for a in args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"失败({proc.returncode}): {' '.join(map(str, args))}\n{proc.stderr[:400]}")


def run_flip(args: list[str], timeout: int = 1200) -> float | None:
    """跑一次并取 teacher_flip_rate（输出里的 Windows 路径不是合法 JSON 转义，不能用 json.loads）。"""
    proc = subprocess.run([str(a) for a in args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"失败({proc.returncode}): {' '.join(map(str, args))}\n{proc.stderr[:400]}")
    match = re.search(r'"teacher_flip_rate":\s*([0-9.]+)', proc.stdout or "")
    return float(match.group(1)) if match else None


def corpus_dir(name: str) -> Path:
    return LAB / name


def dump_profile_codes(profile: Path, out: Path) -> None:
    """把 .mossclone 解成 make_prompt 用的 codes.txt（每行 16 个整数）。"""
    raw = profile.read_bytes()
    magic, version, nvq, frames, rate, name_len = struct.unpack_from("<4sIIIII", raw, 0)
    if magic != b"MSCL" or version != 1 or nvq != 16:
        raise SystemExit(f"音色档案格式不对: {profile}")
    off = struct.calcsize("<4sIIIII") + name_len
    codes = np.frombuffer(raw, dtype="<i2", count=frames * 16, offset=off).reshape(frames, 16)
    out.write_text("\n".join(" ".join(str(int(v)) for v in row) for row in codes) + "\n", encoding="utf-8")


def ensure_corpus(name: str) -> dict:
    spec = CORPORA[name]
    d = corpus_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    prompt = d / "prompt.txt"
    codes = d / "codes.txt"

    if not prompt.exists():
        cmd = [PY, ROOT / "tools" / "make_prompt.py", "--text", spec["text"],
               "--voice", spec["voice"], "--out", prompt]
        if "profile" in spec:
            vc = d / "voice_codes.txt"
            dump_profile_codes(Path(spec["profile"]), vc)
            cmd += ["--codes", vc]
        print(f"[setup] prompt {name}", flush=True)
        run_quiet(cmd)

    frames = spec["frames"]
    if not codes.exists():
        print(f"[setup] codes {name} ({frames} 帧, f16 贪心)", flush=True)
        run_quiet([EXE, "--weights", F16, "--prompt", prompt, "--frames", frames, "--greedy",
                   "--min-frames", frames, "--threads", "4", "--out", codes])

    ref_f16 = d / "ref_f16.bin"
    st_f16 = d / "state_f16.bin"
    ref_q8 = d / "ref_q8.bin"
    st_q8 = d / "state_q8.bin"
    if not ref_f16.exists() or not st_f16.exists():
        print(f"[setup] f16 参考 {name}", flush=True)
        run_quiet([EXE, "--weights", F16, "--prompt", prompt, "--frames", frames, "--threads", "4",
                   "--force-codes", codes, "--dump-logits", ref_f16, "--save-state", st_f16])
    if not ref_q8.exists() or not st_q8.exists():
        print(f"[setup] q8 参考 {name}", flush=True)
        run_quiet([EXE, "--weights", Q8, "--prompt", prompt, "--frames", frames, "--threads", "4",
                   "--force-codes", codes, "--dump-logits", ref_q8, "--save-state", st_q8])
    return {"prompt": prompt, "codes": codes, "frames": frames, "ref_f16": ref_f16, "ref_q8": ref_q8,
            "state_f16": st_f16, "state_q8": st_q8}


def export_variant(group: dict) -> Path:
    """导出一份「只有该组是 Q4、其余 Q8」的权重。"""
    name = group["name"]
    wpath = LAB / "w" / f"q4_{name}.moss"
    if wpath.exists():
        return wpath
    plan = {"default": "q8_0_4x4",
            "overrides": [{"pattern": p, "type": "q4_0_4x4"} for p in group["patterns"]]}
    plans = LAB / "plans"
    plans.mkdir(parents=True, exist_ok=True)
    plan_path = plans / f"q4_{name}.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"[export] {name}", flush=True)
    run_quiet([PY, ROOT / "tools" / "export_weights.py", "--ckpt", CKPT,
               "--out", wpath, "--plan-file", plan_path])
    return wpath


def softmax(x: np.ndarray, t: float) -> np.ndarray:
    z = x.astype(np.float64) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_logits(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    return raw.reshape(-1, 1024)


def compare(ref: Path, test: Path, temperature: float = 0.8, k: int = 25) -> dict:
    a = load_logits(ref)
    b = load_logits(test)
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    p = softmax(a, temperature)
    q = softmax(b, temperature)
    kl = np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=1)
    top1 = (a.argmax(axis=1) == b.argmax(axis=1)).astype(np.float64)
    ia = np.argpartition(-a, k, axis=1)[:, :k]
    ib = np.argpartition(-b, k, axis=1)[:, :k]
    overlap = np.array([len(set(ia[i].tolist()) & set(ib[i].tolist())) / k for i in range(n)])
    return {"steps": int(n), "kl_mean": float(kl.mean()), "kl_p95": float(np.percentile(kl, 95)),
            "top1": float(top1.mean()), "topk_overlap": float(overlap.mean())}


def sweep_one(group: dict, corpus: str, ref: dict, wpath: Path, workdir: Path,
              ref_bin: Path) -> dict:
    name, frames = group["name"], ref["frames"]
    prompt, codes = ref["prompt"], ref["codes"]
    ref_q8 = ref["ref_q8"]

    # 1) 该组 Q4 的 prefill 状态
    st_cand = workdir / f"state_{name}.bin"
    run_quiet([EXE, "--weights", wpath, "--prompt", prompt, "--frames", "1", "--threads", "4",
               "--save-state", st_cand])

    # 2) 只降 decode：Q8 prefill + 该组 Q4 decode
    dec_only = workdir / f"dec_{name}.bin"
    dec_flip = run_flip([EXE, "--weights", wpath, "--prompt", prompt, "--frames", frames, "--threads", "4",
                         "--greedy", "--force-codes", codes, "--load-state", ref["state_q8"], "--ignore-state-fp",
                         "--dump-logits", dec_only])

    # 3) 只降 prefill：该组 Q4 prefill + Q8 decode
    pre_only = workdir / f"pre_{name}.bin"
    pre_flip = run_flip([EXE, "--weights", Q8, "--prompt", prompt, "--frames", frames, "--threads", "4",
                         "--force-codes", codes, "--load-state", st_cand, "--ignore-state-fp",
                         "--dump-logits", pre_only])

    st_cand.unlink(missing_ok=True)
    return {
        "group": name, "corpus": corpus,
        # 相对 f16 的绝对精度
        "decode_only": compare(ref_bin, dec_only),
        "prefill_only": compare(ref_bin, pre_only),
        # 相对 Q8 生产的边际损失（决策用这个）
        "decode_marginal": compare(ref_q8, dec_only),
        "prefill_marginal": compare(ref_q8, pre_only),
        "decode_flip": dec_flip,
        "prefill_flip": pre_flip,
    }


def cmd_setup(args) -> None:
    names = args.corpora.split(",") if args.corpora else list(CORPORA)
    for name in names:
        ensure_corpus(name)


def cmd_sweep(args) -> None:
    corpora = args.corpora.split(",") if args.corpora else ["zh_mid", "zh_long"]
    groups = group_specs()
    if args.groups:
        want = set(args.groups.split(","))
        groups = [g for g in groups if g["name"] in want]

    results_path = LAB / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    done = {(r["group"], r["corpus"]) for r in results}

    refs = {name: ensure_corpus(name) for name in corpora}
    for idx, group in enumerate(groups, 1):
        pending = [c for c in corpora if (group["name"], c) not in done]
        if not pending:
            continue
        wpath = export_variant(group)
        for corpus in pending:
            ref = refs[corpus]
            workdir = corpus_dir(corpus)
            row = sweep_one(group, corpus, ref, wpath, workdir, ref["ref_f16"])
            saved = (group["weights"] * Q8_BYTES_PER_WEIGHT * group["per_frame"] -
                     group["weights"] * Q4_BYTES_PER_WEIGHT * group["per_frame"]) / 1e6
            row["saved_mb_per_frame"] = round(saved, 2)
            row["weights"] = group["weights"]
            row["per_frame"] = group["per_frame"]
            row["label"] = group["label"]
            results.append(row)
            print(f"[{idx}/{len(groups)}] {group['name']:7s} {corpus:8s} "
                  f"dec dKL {row['decode_marginal']['kl_mean']:.5f} top1 {row['decode_marginal']['top1']:.3f} | "
                  f"pre dKL {row['prefill_marginal']['kl_mean']:.5f} top1 {row['prefill_marginal']['top1']:.3f} | "
                  f"-{saved:.1f} MB", flush=True)
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        wpath.unlink(missing_ok=True)
    print(f"done: {len(results)} rows -> {results_path}")


def cmd_report(args) -> None:
    results_path = LAB / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not results:
        print("没有结果")
        return

    rows: dict[str, dict] = {}
    for r in results:
        if "decode_marginal" not in r:
            continue
        agg = rows.setdefault(r["group"], {"label": r["label"], "saved": r["saved_mb_per_frame"],
                                           "dec_kl": [], "dec_top1": [], "dec_abs": [],
                                           "pre_kl": [], "pre_top1": [], "pre_abs": [],
                                           "dec_flip": [], "pre_flip": []})
        agg["dec_kl"].append(r["decode_marginal"]["kl_mean"])
        agg["dec_top1"].append(r["decode_marginal"]["top1"])
        agg["dec_abs"].append(r["decode_only"]["kl_mean"])
        agg["pre_kl"].append(r["prefill_marginal"]["kl_mean"])
        agg["pre_top1"].append(r["prefill_marginal"]["top1"])
        agg["pre_abs"].append(r["prefill_only"]["kl_mean"])
        if r.get("decode_flip") is not None:
            agg["dec_flip"].append(r["decode_flip"])
        if r.get("prefill_flip") is not None:
            agg["pre_flip"].append(r["prefill_flip"])

    table = []
    for name, a in rows.items():
        dec_kl = float(np.mean(a["dec_kl"]))
        pre_kl = float(np.mean(a["pre_kl"]))
        table.append({
            "group": name, "label": a["label"], "saved_mb": a["saved"],
            "dec_kl": dec_kl, "dec_top1": float(np.mean(a["dec_top1"])),
            "dec_abs": float(np.mean(a["dec_abs"])),
            "pre_kl": pre_kl, "pre_top1": float(np.mean(a["pre_top1"])),
            "pre_abs": float(np.mean(a["pre_abs"])),
            "dec_flip": float(np.mean(a["dec_flip"])) if a["dec_flip"] else None,
            "pre_flip": float(np.mean(a["pre_flip"])) if a["pre_flip"] else None,
            "score_dec": (a["saved"] / 263.0 * 100) / max(dec_kl / 0.004, 0.01),
            "score_pre": (a["saved"] / 263.0 * 100) / max(pre_kl / 0.004, 0.01),
        })
    table.sort(key=lambda r: -(r["score_dec"] + r["score_pre"]))
    header = (f"{'group':8s}{'label':22s}{'MB/frame':>9s}{'dec dKL':>9s}{'decTop1':>8s}"
              f"{'decAbs':>8s}{'pre dKL':>9s}{'preTop1':>8s}{'preAbs':>8s}{'S_dec':>7s}{'S_pre':>7s}")
    print(header)
    for r in table:
        print(f"{r['group']:8s}{r['label']:22s}{r['saved_mb']:9.1f}{r['dec_kl']:9.5f}{r['dec_top1']:8.3f}"
              f"{r['dec_abs']:8.4f}{r['pre_kl']:9.5f}{r['pre_top1']:8.3f}{r['pre_abs']:8.4f}"
              f"{r['score_dec']:7.1f}{r['score_pre']:7.1f}")
    (LAB / "report.json").write_text(json.dumps(table, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_setup = sub.add_parser("setup")
    p_setup.add_argument("--corpora", default=None)
    p_setup.set_defaults(func=cmd_setup)

    p_sweep = sub.add_parser("sweep")
    p_sweep.add_argument("--corpora", default=None)
    p_sweep.add_argument("--groups", default=None)
    p_sweep.set_defaults(func=cmd_sweep)

    p_report = sub.add_parser("report")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
