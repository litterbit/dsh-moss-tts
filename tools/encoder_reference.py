"""编码器的 torch 参考实现（克隆链路对拍用）。

按 MOSS-Audio-Tokenizer 的 modeling 代码逐段复现：
    waveform -> 交织 + 补零 -> patch240 -> PT1 -> patch2 -> PT2 -> patch2 -> PT3
    -> patch2 -> PT4 -> patch4 -> quantizer(input_proj + 16×LFQ) -> codes

用法：
    python tools/encoder_reference.py --src <safetensors> --wav <x.wav> --out codes.txt [--dump dir]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from export_decoder import load_safetensors, merge_weight_norm  # noqa: E402

STAGES = (
    # (patch, in_dim, out_dim, layers, window)
    (240, 240, 384, 4, 1600),
    (2, 768, 384, 2, 1200),
    (2, 768, 384, 2, 800),
    (2, 768, 192, 4, 500),
)


def linear(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return x @ w.t()


def layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + 1e-5) * w + b


def rotate_half_style(q: torch.Tensor, pos: torch.Tensor, max_period: float = 10000.0) -> torch.Tensor:
    # q: [T, H, D]；interleaved RoPE
    t, h, d = q.shape
    ds = torch.arange(d // 2, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / d))
    ang = pos.float().view(t, 1, 1) * freqs.view(1, 1, -1)
    c, s = torch.cos(ang), torch.sin(ang)
    qr = q.view(t, h, d // 2, 2)[..., 0]
    qi = q.view(t, h, d // 2, 2)[..., 1]
    return torch.stack([qr * c - qi * s, qr * s + qi * c], dim=-1).view(t, h, d)


def attention(qkv: torch.Tensor, heads: int, head_dim: int, window: int, rope: str = "interleaved",
              causal: bool = True) -> torch.Tensor:
    t, _ = qkv.shape
    q, k, v = qkv[:, : heads * head_dim], qkv[:, heads * head_dim : 2 * heads * head_dim], qkv[:, 2 * heads * head_dim :]
    d = heads * head_dim
    q = q.view(t, heads, head_dim)
    k = k.view(t, heads, head_dim)
    v = v.view(t, heads, head_dim)
    pos = torch.arange(t)
    if rope == "half":
        def apply_half(qq: torch.Tensor) -> torch.Tensor:
            ds = torch.arange(d // 2, dtype=torch.float32)
            freqs = torch.exp(ds * (-math.log(10000.0) * 2 / d))
            ang = pos.float().view(t, 1, 1) * freqs.view(1, 1, -1)
            c, s = torch.cos(ang), torch.sin(ang)
            q1, q2 = qq[..., : d // 2], qq[..., d // 2 :]
            return torch.cat([q1 * c - q2 * s, q1 * s + q2 * c], dim=-1)

        q = apply_half(q)
        k = apply_half(k)
    else:
        q = rotate_half_style(q, pos)
        k = rotate_half_style(k, pos)
    delta = pos.view(t, 1) - pos.view(1, t)
    mask = (delta >= 0) if causal else torch.ones((t, t), dtype=torch.bool)
    if window > 0 and causal:
        mask = mask & (delta < window)
    scores = torch.einsum("thd,shd->hts", q, k) * (head_dim**-0.5)
    scores = scores.masked_fill(~mask.view(1, t, t), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hts,shd->thd", probs, v)
    return out.reshape(t, d)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dump", default=None)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--window-mode", default="config", choices=["config", "full"])
    parser.add_argument("--rope", default="interleaved", choices=["interleaved", "half"])
    parser.add_argument("--quant-mode", default="norm", choices=["norm", "norm_enc", "l2", "raw", "enc_only"])
    parser.add_argument("--causal", type=int, default=1)
    parser.add_argument("--dump-tensors", default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    st = load_safetensors(Path(args.src))
    W = {k: torch.from_numpy(v) for k, v in st.items()}

    wav, sr = torchaudio.load(args.wav)
    if sr != 48000:
        wav = torchaudio.functional.resample(wav, sr, 48000)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    if args.seconds > 0:
        wav = wav[:, : int(args.seconds * 48000)]
    per_channel = wav.shape[-1]
    inter = wav.t().reshape(1, 1, -1)  # L,R,L,R...
    total = inter.shape[-1]
    # ONNX 导出用的是 pad = 3840 - (len % 3840)：对齐时也补一整帧，这里照抄以便逐张量对拍
    pad_per_channel = 3840 - (per_channel % 3840)
    pad = pad_per_channel * 2
    if pad:
        inter = torch.nn.functional.pad(inter, (0, pad))

    x = inter  # [1, 1, N]
    dumps = {}

    for si, (patch, in_dim, out_dim, layers, window) in enumerate(STAGES):
        p = f"encoder.{si * 2 + 1}"
        # patch（下采样）
        b, c, n = x.shape
        x = x.reshape(b, c, -1, patch).permute(0, 1, 3, 2).reshape(b, c * patch, -1)
        x = x.reshape(-1, x.shape[-1]).t().contiguous()  # [T, C]
        # in_proj
        y = linear(W[f"{p}.input_proj.weight"], x)
        for li in range(layers):
            lp = f"{p}.transformer.layers.{li}"
            h = layernorm(y, W[f"{lp}.norm1.weight"], W[f"{lp}.norm1.bias"])
            a = linear(W[f"{lp}.self_attn.in_proj.weight"], h).contiguous()
            a = attention(a, 4, 64, 0 if args.window_mode == "full" else window, args.rope, args.causal)
            a = linear(W[f"{lp}.self_attn.out_proj.weight"], a)
            y = y + W[f"{lp}.layer_scale_1.scale"] * a
            h = layernorm(y, W[f"{lp}.norm2.weight"], W[f"{lp}.norm2.bias"])
            f = linear(W[f"{lp}.ffn.0.weight"], h)
            f = torch.nn.functional.gelu(f, approximate="none")
            f = linear(W[f"{lp}.ffn.2.weight"], f)
            y = y + W[f"{lp}.layer_scale_2.scale"] * f
        x = linear(W[f"{p}.output_proj.weight"], y).t().unsqueeze(0)  # [1, C, T]
        dumps[f"stage{si}"] = (x.mean().item(), x.abs().max().item(), x.shape)
        if args.dump_tensors:
            import numpy as _np
            _np.save(Path(args.dump_tensors) / f"stage{si}.npy", x.squeeze(0).transpose(0, 1).numpy())

    # 最后的 patch4
    b, c, n = x.shape
    x = x.reshape(b, c, -1, 4).permute(0, 1, 3, 2).reshape(b, c * 4, -1)
    x = x.reshape(-1, x.shape[-1]).t().contiguous()  # [T, 768]

    # quantizer
    z = linear(torch.from_numpy(merge_weight_norm(st, "quantizer.input_proj")), x)
    z = z + torch.from_numpy(st["quantizer.input_proj.bias"])
    residual = z.clone()
    nq = 16
    codes = torch.zeros(x.shape[0], nq, dtype=torch.long)
    for i in range(nq):
        qp = f"quantizer.quantizers.{i}"
        in_w = torch.from_numpy(merge_weight_norm(st, f"{qp}.in_proj"))
        out_w = torch.from_numpy(merge_weight_norm(st, f"{qp}.out_proj"))
        in_b = torch.from_numpy(st[f"{qp}.in_proj.bias"])
        out_b = torch.from_numpy(st[f"{qp}.out_proj.bias"])
        cb = torch.from_numpy(st[f"{qp}.codebook.weight"])
        e = linear(in_w, residual) + in_b
        if args.quant_mode == "l2":
            idx = ((e * e).sum(-1, keepdim=True) - 2 * e @ cb.t() + (cb * cb).sum(-1).view(1, -1)).argmin(dim=-1)
            cbn = cb
        elif args.quant_mode == "enc_only":
            e = torch.nn.functional.normalize(e, dim=-1)
            dist = (e * e).sum(-1, keepdim=True) - 2 * e @ cb.t() + (cb * cb).sum(-1).view(1, -1)
            idx = (-dist).argmax(dim=-1)
            cbn = cb
        elif args.quant_mode == "raw":
            idx = (e @ cb.t()).argmax(dim=-1)
            cbn = cb
        else:
            e = torch.nn.functional.normalize(e, dim=-1)
            cbn = torch.nn.functional.normalize(cb, dim=-1)
            idx = (e @ cbn.t()).argmax(dim=-1)
        codes[:, i] = idx
        # 关键：最近邻在**归一化**空间里找，但重建残差用的是**原始**码本向量
        rec = linear(out_w, cb[idx]) + out_b
        residual = residual - rec

    valid = min(codes.shape[0], per_channel // 3840)
    out = Path(args.out)
    out.write_text("\n".join(" ".join(str(int(v)) for v in row) for row in codes[:valid].tolist()) + "\n",
                   encoding="utf-8")
    if args.dump:
        Path(args.dump).write_text(json.dumps({k: v for k, v in dumps.items()}, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "frames": valid, "tokens": int(codes.shape[0]),
                      "stages": {k: [round(v[0], 5), round(v[1], 4)] for k, v in dumps.items()}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
