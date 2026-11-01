"""解码器语义参考实现（torch fp32），用于在写 C++ 之前验证结构理解。

对拍对象：ONNX `moss_audio_tokenizer_decode_full.onnx`。

注意：本机 conda numpy 的 BLAS 在 matmul 时会崩（0xc06d007f），所以这里用 torch。

用法：
    python tools/decoder_reference.py --codes runs/cpp/codes_q8x4_67.txt [--save ref.pt]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TOKENIZER = Path(r"D:\workplace\audio\github\MOSS-Audio-Tokenizer-Nano")
ONNX_DIR = Path(r"D:\workplace\audio\github\MOSS-Audio-Tokenizer-Nano-ONNX")


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        base = 8 + n
        out: dict[str, torch.Tensor] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            with path.open("rb") as f:
                f.seek(base + start)
                raw = f.read(end - start)
            arr = np.frombuffer(raw, dtype=np.float32).reshape(tuple(info["shape"])).copy()
            out[name] = torch.from_numpy(arr)
    return out


def merge_wn(t: dict[str, torch.Tensor], p: str) -> torch.Tensor:
    g = t[p + ".parametrizations.weight.original0"]
    v = t[p + ".parametrizations.weight.original1"]
    norm = v.reshape(v.shape[0], -1).norm(dim=1).reshape(-1, 1, 1).clamp_min(1e-12)
    return (g * v / norm).reshape(v.shape[0], v.shape[1])


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="none")


def rope_apply(x: torch.Tensor, positions: torch.Tensor, max_period: float = 10000.0) -> torch.Tensor:
    """x: [H, T, D]，interleaved RoPE。"""
    h, t, d = x.shape
    half = d // 2
    ds = torch.arange(half, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2.0 / d))       # [half]
    ang = positions.float().view(-1, 1) * freqs.view(1, -1)          # [T, half]
    cos, sin = ang.cos()[None], ang.sin()[None]
    xr = x.reshape(h, t, half, 2)
    r, i = xr[..., 0], xr[..., 1]
    out = torch.stack([r * cos - i * sin, r * sin + i * cos], dim=-1)
    return out.reshape(h, t, d)


def patch(x: torch.Tensor, size: int) -> torch.Tensor:
    c, t = x.shape
    d = c // size
    return x.reshape(d, size, t).permute(0, 2, 1).reshape(d, t * size)


class Decoder:
    def __init__(self, tensors: dict[str, torch.Tensor], heads: int = 4) -> None:
        self.t = tensors
        self.heads = heads

    def _attention(self, x: torch.Tensor, name: str) -> torch.Tensor:
        t, c = x.shape
        dim = c // self.heads
        qkv = x @ self.t[f"{name}.self_attn.in_proj.weight"].T          # [T,3C]
        qkv = qkv.reshape(t, 3, self.heads, dim)
        q = qkv[:, 0].permute(1, 0, 2).contiguous()                     # [H,T,D]
        k = qkv[:, 1].permute(1, 0, 2).contiguous()
        v = qkv[:, 2].permute(1, 0, 2).contiguous()

        pos = torch.arange(t, dtype=torch.float32)
        q = rope_apply(q, pos)
        k = rope_apply(k, pos)

        scores = torch.einsum("htd,hud->htu", q, k) / math.sqrt(dim)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask[None], float("-inf"))
        prob = torch.softmax(scores, dim=-1)
        out = torch.einsum("htu,hud->htd", prob, v).permute(1, 0, 2).reshape(t, c)
        return out @ self.t[f"{name}.self_attn.out_proj.weight"].T

    def _layer(self, x: torch.Tensor, name: str) -> torch.Tensor:
        t = self.t
        xn = F.layer_norm(x, (x.shape[-1],), t[f"{name}.norm1.weight"], t[f"{name}.norm1.bias"], 1e-5)
        x = x + t[f"{name}.layer_scale_1.scale"] * self._attention(xn, name)
        xn = F.layer_norm(x, (x.shape[-1],), t[f"{name}.norm2.weight"], t[f"{name}.norm2.bias"], 1e-5)
        f = gelu(xn @ t[f"{name}.ffn.0.weight"].T) @ t[f"{name}.ffn.2.weight"].T
        return x + t[f"{name}.layer_scale_2.scale"] * f

    def _stage(self, x: torch.Tensor, stage: int, layers: int) -> torch.Tensor:
        p = f"decoder.{stage}"
        y = x.T @ self.t[f"{p}.input_proj.weight"].T
        for n in range(layers):
            y = self._layer(y, f"{p}.transformer.layers.{n}")
        return (y @ self.t[f"{p}.output_proj.weight"].T).T

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        nq, t = codes.shape
        z = torch.zeros(512, t)
        for i in range(nq):
            emb = self.t[f"quantizer.quantizers.{i}.codebook.weight"][codes[i]]
            w = merge_wn(self.t, f"quantizer.quantizers.{i}.out_proj")
            b = self.t[f"quantizer.quantizers.{i}.out_proj.bias"]
            z = z + (emb @ w.T + b).T
        wo = merge_wn(self.t, "quantizer.output_proj")
        bo = self.t["quantizer.output_proj.bias"]
        return wo @ z + bo[:, None]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        x = self.decode_codes(codes)
        x = patch(x, 4)
        x = self._stage(x, 1, 4)
        for stage in (3, 5):
            x = patch(x, 2)
            x = self._stage(x, stage, 2)
        x = patch(x, 2)
        x = self._stage(x, 7, 4)
        x = patch(x, 240)                       # [1, T*7680]
        inter = x.reshape(-1)
        return torch.stack([inter[0::2], inter[1::2]], dim=0)


def onnx_decode(codes: torch.Tensor) -> torch.Tensor:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_DIR / "moss_audio_tokenizer_decode_full.onnx"))
    nq, t = codes.shape
    # ONNX 侧布局是 [B, T, nq]
    audio, lengths = sess.run(
        None,
        {"audio_codes": codes.numpy().T[None].astype(np.int32),
         "audio_code_lengths": np.array([t], dtype=np.int32)},
    )
    n = int(np.asarray(lengths).reshape(-1)[0])
    return torch.from_numpy(audio[0, :, :n].copy())


def snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    n = min(ref.shape[-1], test.shape[-1])
    a, b = ref[..., :n].double(), test[..., :n].double()
    noise = a - b
    return float(10.0 * torch.log10((a * a).sum() / (noise * noise).sum().clamp_min(1e-20)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", required=True)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)

    rows = [[int(x) for x in line.split()] for line in Path(args.codes).read_text().splitlines() if line.strip()]
    codes = torch.tensor(rows, dtype=torch.long).T                      # [16, T]

    dec = Decoder(load_safetensors(TOKENIZER / "model-00001-of-00001.safetensors"))
    mine = dec.decode(codes)
    ref = onnx_decode(codes)

    result = {
        "frames": int(codes.shape[1]),
        "mine_shape": list(mine.shape),
        "onnx_shape": list(ref.shape),
        "snr_db": round(snr_db(ref, mine), 2),
        "peak_mine": round(float(mine.abs().max()), 4),
        "peak_onnx": round(float(ref.abs().max()), 4),
    }
    print(json.dumps(result, ensure_ascii=False))
    if args.save:
        torch.save(mine, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
