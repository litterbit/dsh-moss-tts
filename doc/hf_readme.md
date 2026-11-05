---
license: apache-2.0
language:
  - zh
  - en
tags:
  - text-to-speech
  - tts
  - cpu-inference
  - quantized
  - cpp
pretty_name: MOSS-TTS-Nano-100M CPU 权重（int8 4×4 / fp16）
---

# MOSS-TTS-Nano-100M · CPU 权重包

**English.** Quantized inference weights for the standalone C++/CPU engine
[dsh-moss-tts](https://github.com/litterbit/dsh-moss-tts). Derived from **MOSS-TTS-Nano-100M**
(Apache-2.0): the LLM is cast to int8 (`Q8_0`, repacked 4×4), the audio decoder/encoder to fp16,
and every tensor is rewritten into the project's `.moss` container. **Nothing is re-trained** —
architecture, parameter values and the public prompt template are unchanged; only the numeric
format differs. These files cannot be loaded by `transformers`; they need the engine.
This repo mirrors the `models/` directory of the release zip and matches
[`scripts/model_lock.json`](https://github.com/litterbit/dsh-moss-tts/blob/main/scripts/model_lock.json)
byte for byte.

## 这是什么

- 一个 100 M 参数的中英双语 TTS：一帧 80 ms、16 路 VQ（每路 1024 码）、输出 48 kHz 立体声。
- 结构：12 层 GPT-2 风格全局模型（d=768，RoPE）+ 1 层 local 头逐通道出码；音频解码器把
  16×T 的码上采样成 48 kHz（4+2+2+4 层，patch 4/2/2/2/240）。
- 文本前端自带分词（sentencepiece 词表已导出为 `tokenizer.mossvocab`，427 KB），不依赖 Python。

## 文件

<!--FILES-->

`manifest.json` 是同一份清单的副本，引擎读它得到 `model_id` / `model_revision`；
`voices/*.mossclone` 是 18 个内置音色（参考音频经编码器得到的 VQ 码，每个几 KB）。

## 用法

两条路，任选一条。

1. **不想折腾**：下 [发布包](https://github.com/litterbit/dsh-moss-tts/releases/latest)
   （`MossTTS-win-x64-full.zip`，约 159 MB，程序 + 权重 + 音色），解压后双击 `moss_tts.exe`。
2. **只要权重**：把这个仓库下到任意目录当 `models/` 用：

   ```bash
   git clone https://huggingface.co/litterbit/moss-tts-nano-moss models
   moss_tts.exe --models models --text "你好，世界。" --voice Junhao --out hello.wav
   moss_tts.exe --models models            # 或者不带参数：起本地网页试听台
   ```

   也可以只下 [lite 包](https://github.com/litterbit/dsh-moss-tts/releases/latest) 再把 `models/` 放进去。

引擎的查找顺序是 `--models` → exe 同级的 `models/` → `%LOCALAPPDATA%\MossTTS\models`，
所以把本仓库解到 exe 旁边的 `models/` 里最省事。

## 校验

每个文件的 sha256 都在上表里，也写在 `manifest.json`。核对：

```bash
sha256sum -c sha256sums.txt     # 从发布页取（Windows: certutil -hashfile <file> SHA256）
```

引擎自己也会校验吗？`--selftest` 会加载全部权重并合成一段做峰值检查；发布流水线在下载后
逐个文件比对 `scripts/model_lock.json`（仓库里的信任根，不是"下完再生成"）。

## 量化与实测

| 部分 | 精度 | 体积 |
|---|---|---|
| LLM 权重 | int8 `Q8_0`（32 元素一组 scale，4 行一组重排） | 119.3 MB |
| LLM 激活 | 解码段 fp32（prefill 仍 int8） | — |
| 音频解码器 / 编码器 | fp16 | 22.3 / 21.8 MB |
| 分词词表 | 原始 | 427 KB |

精度（teacher forcing，和 fp16 参考比，同一段 in-domain 语料）：

| 配置 | code 翻转 | KL | 说明 |
|---|---|---|---|
| int8 权重 + fp32 解码激活（发布默认） | 5.2% | 0.00145 | 听不出差别 |
| 激活也量化成 int8 | 8.7% | 0.0042 | 备选，快 15% |
| 权重再降到 Q4 | 64~86% | 0.44~1.16 | **不可用**，响度都会变 |

速度（i5-14600KF，4+6 线程，纯粹在 CPU 上）：LLM 7.4 ms/帧、解码器 6.2 ms/帧，
5.12 s 语音**首块 93~114 ms、总共 764 ms**（RTF 0.149）；16.7 ms/帧是关闭 AVX2 的标量回退。

## 音色

- 18 个内置音色：`Junhao`、`Xiaoyu`、`Ava`、`Trump`…… 都是上游自带参考音频编码出来的。
- 自己的音色：给一段 5~15 秒 WAV（PCM），网页上点"创建音色"或
  `moss_clone.exe --encoder models/encoder_f16.moss --in ref.wav --out voices/me.mossclone`。
  产物只有几 KB，和内置音色物理隔离存放，升级不会覆盖。

## 许可与归属

- 上游：**MOSS-TTS-Nano-100M**，Apache-2.0。
- 本仓库是它的**量化/重排产物**，同样适用 Apache-2.0；未对结构、训练数据或参数取值做再训练。
- 本项目与 MOSS 模型作者无隶属关系，未获其背书。完整声明见仓库根的 `NOTICE` / `LICENSE`。

## 相关链接

- 引擎与网页：<https://github.com/litterbit/dsh-moss-tts>
- 下载与发版说明：<https://github.com/litterbit/dsh-moss-tts/releases/latest>
- HTTP / stdio 接口：<https://github.com/litterbit/dsh-moss-tts/blob/main/doc/api.md>
