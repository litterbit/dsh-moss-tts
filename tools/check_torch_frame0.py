"""硬回归：原始 PyTorch 模型对已验 prompt 的贪心帧 0 必须与 C++/ONNX 逐位相同。

``runs/probe/p_short.txt`` 是 C++ 引擎用的 180 行 prompt，``ours_g_short.txt`` 是对应
的 C++ 贪心 codes（其帧 0 已与官方 ONNX 16/16 对拍过）。这里把同一份 rows 直接喂给
PyTorch 模型，绕过分词/编码器/prompt 构造，只验模型数学。

    python tools/check_torch_frame0.py
    python tools/check_torch_frame0.py --prompt-rows runs/probe/p_short.txt --ref runs/probe/ours_g_short.txt

帧 0 不一致 → 退出码 1。帧 1 起允许少量差异（C++ Q8 量化本底会级联）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from dsh_moss_tts import torch_ref  # noqa: E402
from dsh_moss_tts.paths import RUNS_DIR, TTS_CKPT_DIR  # noqa: E402

N_VQ = 16


def _load_rows(path: Path, width: int) -> np.ndarray:
    rows = np.loadtxt(path, dtype=np.int64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.shape[1] != width:
        raise SystemExit(f"{path}: 期望每行 {width} 个整数，实际 {rows.shape[1]}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-rows", default=str(RUNS_DIR / "probe" / "p_short.txt"))
    parser.add_argument("--ref", default=str(RUNS_DIR / "probe" / "ours_g_short.txt"))
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--ckpt", default=str(TTS_CKPT_DIR))
    parser.add_argument("--attn", default="sdpa", choices=("sdpa", "eager"))
    args = parser.parse_args(argv)

    prompt_path, ref_path = Path(args.prompt_rows), Path(args.ref)
    for path in (prompt_path, ref_path):
        if not path.is_file():
            raise SystemExit(f"找不到 {path}")

    rows = _load_rows(prompt_path, N_VQ + 1)
    reference = _load_rows(ref_path, N_VQ)  # C++ 贪心 codes，每行 16 个
    torch, model = torch_ref.load_tts(args.ckpt, attn_implementation=args.attn)
    print(f"prompt rows={rows.shape[0]} attn={args.attn}")

    input_ids = torch.from_numpy(rows).unsqueeze(0)
    attention_mask = torch.ones(1, rows.shape[0], dtype=torch.bool)
    with torch.no_grad():
        frames = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_frames=args.frames,
            do_sample=False,
        ).audio_token_ids[0].tolist()

    frames = np.asarray(frames, dtype=np.int64)
    print(f"generated frames={frames.shape[0]}")
    frame0_equal = int((frames[0] == reference[0]).sum())
    print(f"frame0 match {frame0_equal}/{N_VQ}")
    print(f"  torch={frames[0].tolist()}")
    print(f"  ref  ={reference[0].tolist()}")

    for index in range(1, min(frames.shape[0], reference.shape[0])):
        mismatch = int((frames[index] != reference[index]).sum())
        print(f"frame{index} mismatch {mismatch}/{N_VQ} (量化本底会级联，仅供参考)")

    if frame0_equal != N_VQ:
        print("FAIL: 帧 0 与 C++/ONNX 不一致，模型数学路径有问题")
        return 1
    print("PASS: 帧 0 与 C++/ONNX 逐位相同")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
