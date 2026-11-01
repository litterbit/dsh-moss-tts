"""项目里所有路径的唯一来源。"""

from __future__ import annotations

import os
from pathlib import Path

# dsh-moss-tts/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 原样复制的上游参考实现（只读，见 vendor/README.md）
VENDOR_DIR = PROJECT_ROOT / "vendor" / "moss_tts_nano_onnx"

# 权重体积大且跨项目共用，不放进仓库；目录根下要有 browser_poc_manifest.json
MODELS_DIR = Path(os.environ.get("MOSS_MODELS_DIR", r"D:\workplace\audio\github"))

# 原始 PyTorch checkpoint（含 modeling_*.py、config.json、pytorch_model.bin）
TTS_CKPT_DIR = Path(os.environ.get("MOSS_TTS_CKPT_DIR", MODELS_DIR / "MOSS-TTS-Nano-100M"))
CODEC_CKPT_DIR = Path(os.environ.get("MOSS_CODEC_CKPT_DIR", MODELS_DIR / "MOSS-Audio-Tokenizer-Nano"))
# 上游仓库（只用 assets/audio 里的参考音频，不 import 它的代码）
TTS_REPO_DIR = Path(os.environ.get("MOSS_TTS_REPO_DIR", MODELS_DIR / "MOSS-TTS-Nano"))

# 产物目录：放在项目内，不污染工作区根目录
# （注意：沙箱进程是低完整性级别，项目目录需要有 Low 完整性标签才能从沙箱写入，
#  一次性修复见 AGENTS.md「已知坑 1」）
RUNS_DIR = Path(os.environ.get("MOSS_RUNS_DIR", PROJECT_ROOT / "runs"))


def run_dir(name: str) -> Path:
    """runs/<name>/，不存在则创建。"""
    path = RUNS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
