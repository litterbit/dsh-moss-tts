"""从本地文件加载原始 MOSS-TTS-Nano / MOSS-Audio-Tokenizer-Nano PyTorch 权重。

为什么不用 ``from_pretrained(..., trust_remote_code=True)``：

* transformers 5.x 在 meta device 上构建模型，然后把所有**非持久 buffer** 用
  ``torch.empty_like`` 搬回（``transformers/modeling_utils.py:4793-4797``），
  不会重新初始化。checkpoint 的 ``MossTTSNanoGPT2RotaryEmbedding.inv_freq``
  正是 ``register_buffer(..., persistent=False)``，于是变成未初始化内存
  （实测 max |inv_freq| = 3.16e-11），RoPE 静默退化成恒等变换 —— 每个权重都
  逐位加载正确，音频却全错。
* ``trust_remote_code`` 还要写 HuggingFace 动态模块缓存，Windows 上容易撞 MAX_PATH。

这里改为：把模型目录当成一个临时包直接 import 它自带的 .py，按 ``config.json``
构造模型，再用 ``load_state_dict`` 灌权重。buffer 保留 ``__init__`` 算出的值，
不碰 HF 缓存、不联网、不需要 trust_remote_code。

实测（audio 环境，transformers 5.22.1 / torch 2.13.0，CPU fp32）：
两个模型都 0 missing / 0 unexpected；prefill 末位 hidden 与官方 ONNX
``relL2 = 1.26e-6``，贪心帧 0 codes 与 C++/ONNX 16/16 逐位相同。
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

TTS_PACKAGE = "dsh_moss_tts_ckpt"
CODEC_PACKAGE = "dsh_moss_tts_codec"

_TTS_REQUIRED = (
    "config.json",
    "modeling_moss_tts_nano.py",
    "configuration_moss_tts_nano.py",
    "gpt2_decoder.py",
    "prompting.py",
    "tokenization_moss_tts_nano.py",
    "tokenizer.model",
)
_CODEC_REQUIRED = (
    "config.json",
    "modeling_moss_audio_tokenizer.py",
    "configuration_moss_audio_tokenizer.py",
)


def configure_offline_env() -> None:
    """Defensive: 这套加载器不该触发任何网络，也不该写 HF 缓存。"""
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(key, "1")


def _require_files(directory: Path, names: tuple[str, ...], what: str) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{what} 目录 {directory} 缺少文件: {', '.join(missing)}")


def _ensure_package(name: str, directory: Path) -> types.ModuleType:
    """把模型目录注册成一个可 import 的包（不写任何文件）。"""
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {directory}")
    existing = sys.modules.get(name)
    if existing is None:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        package.__file__ = str(directory / "__init__.py")
        sys.modules[name] = package
        return package
    if str(directory) not in list(getattr(existing, "__path__", [])):
        raise RuntimeError(
            f"包 {name} 已绑定到 {list(getattr(existing, '__path__', []))}，不能再指向 {directory}"
        )
    return existing


def _import_module(package: str, directory: Path, module: str):
    _ensure_package(package, directory)
    return importlib.import_module(f"{package}.{module}")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_pytorch_bin(path: Path) -> dict:
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    return state


def _load_safetensors(directory: Path) -> dict:
    from safetensors.torch import load_file

    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = _read_json(index_path).get("weight_map", {})
        filenames = sorted(set(weight_map.values()))
    else:
        filenames = sorted(path.name for path in directory.glob("*.safetensors"))
    if not filenames:
        raise FileNotFoundError(f"{directory} 里没有 *.safetensors")
    state: dict = {}
    for filename in filenames:
        state.update(load_file(str(directory / filename)))
    return state


def _load_weights(directory: Path) -> dict:
    bin_path = directory / "pytorch_model.bin"
    if bin_path.is_file():
        return _load_pytorch_bin(bin_path)
    return _load_safetensors(directory)


def _load_strict(module, state: dict, what: str) -> None:
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{what} 权重不匹配: missing={list(missing)[:10]} unexpected={list(unexpected)[:10]}"
        )


def assert_rope_ok(model, config=None, rtol: float = 1e-5, atol: float = 1e-7) -> int:
    """校验每个 rotary embedding 的 inv_freq 就是训练时那组频率。

    transformers 5 的 from_pretrained 会把它们变成未初始化内存（≈0），
    RoPE 随之失效；这里显式检查，失败就大声报错，不再静默出噪声。
    """
    import torch

    base = 10000.0
    if config is not None:
        gpt2_config = getattr(config, "gpt2_config", config)
        base = float(getattr(gpt2_config, "rope_base", base) or base)

    checked = 0
    for name, module in model.named_modules():
        inv_freq = getattr(module, "inv_freq", None)
        if inv_freq is None or not torch.is_tensor(inv_freq):
            continue
        dim = int(inv_freq.numel()) * 2
        expected = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        actual = inv_freq.detach().float().cpu().reshape(-1)
        if actual.numel() != expected.numel() or not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            max_err = float((actual - expected).abs().max())
            raise RuntimeError(
                f"{name}.inv_freq 不是训练时的 RoPE 频率（base={base}, max_err={max_err:.3e}）；"
                "多半是 from_pretrained 用 torch.empty_like 重建了非持久 buffer，"
                "请改走 dsh_moss_tts.torch_ref 的直载路径。"
            )
        checked += 1
    return checked


def load_tts(ckpt_dir: str | Path, attn_implementation: str = "sdpa", verify: bool = True):
    """返回 ``(torch, model)``，模型在 CPU fp32、eval 模式。"""
    configure_offline_env()
    import torch

    ckpt = Path(ckpt_dir).resolve()
    _require_files(ckpt, _TTS_REQUIRED, "TTS checkpoint")

    config_module = _import_module(TTS_PACKAGE, ckpt, "configuration_moss_tts_nano")
    modeling_module = _import_module(TTS_PACKAGE, ckpt, "modeling_moss_tts_nano")

    config = config_module.MossTTSNanoConfig(**_read_json(ckpt / "config.json"))
    # config.json 写的是 flash_attention_2；本机没有 flash_attn，显式改掉。
    config.attn_implementation = attn_implementation
    config.local_transformer_attn_implementation = attn_implementation

    model = modeling_module.MossTTSNanoForCausalLM(config)
    model._set_attention_implementation(attn_implementation)
    _load_strict(model, _load_weights(ckpt), f"TTS checkpoint {ckpt}")
    model.eval()

    if verify:
        assert_rope_ok(model, config)
    return torch, model


def load_codec(codec_dir: str | Path):
    configure_offline_env()

    codec = Path(codec_dir).resolve()
    _require_files(codec, _CODEC_REQUIRED, "Audio tokenizer")

    config_module = _import_module(CODEC_PACKAGE, codec, "configuration_moss_audio_tokenizer")
    modeling_module = _import_module(CODEC_PACKAGE, codec, "modeling_moss_audio_tokenizer")

    config = config_module.MossAudioTokenizerConfig(**_read_json(codec / "config.json"))
    model = modeling_module.MossAudioTokenizerModel(config)
    _load_strict(model, _load_safetensors(codec), f"Audio tokenizer {codec}")
    model.eval()
    return model


def load_text_tokenizer(model, ckpt_dir: str | Path):
    ckpt = Path(ckpt_dir).resolve()
    return model._load_text_tokenizer(text_tokenizer_path=str(ckpt / "tokenizer.model"))


def load_all(
    ckpt_dir: str | Path | None = None,
    codec_dir: str | Path | None = None,
    attn_implementation: str = "sdpa",
    with_codec: bool = True,
    verify: bool = True,
):
    """返回 ``(torch, model, text_tokenizer, codec_or_None)``。"""
    if ckpt_dir is None:
        from .paths import TTS_CKPT_DIR

        ckpt_dir = TTS_CKPT_DIR
    if codec_dir is None and with_codec:
        from .paths import CODEC_CKPT_DIR

        codec_dir = CODEC_CKPT_DIR

    torch, model = load_tts(ckpt_dir, attn_implementation=attn_implementation, verify=verify)
    tokenizer = load_text_tokenizer(model, ckpt_dir)
    codec = load_codec(codec_dir) if with_codec else None
    return torch, model, tokenizer, codec
