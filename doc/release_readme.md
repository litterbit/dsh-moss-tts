# MOSS-TTS-Nano CPU 推理（Windows x64，独立可用）

纯 C++ / 无第三方运行时依赖（`dumpbin /dependents` 只有 KERNEL32.dll），CPU 推理，
支持完整音频与流式 PCM 输出，支持输入参考音频做音色克隆。

## 快速开始

```bat
:: 列内置音色
moss_tts.exe --models models --list-voices

:: 合成一句话（48 kHz 立体声 WAV）
moss_tts.exe --models models --text "你好，我是语音助手。" --voice Junhao --out hello.wav

:: 流式输出裸 PCM（s16le / 48 kHz / 立体声），直接管道给播放器
moss_tts.exe --models models --text "这是一段流式输出。" --voice Xiaoyu --pcm-raw - | ffplay -f s16le -ar 48000 -ac 2 -

:: 自检（加载 + 合成 + 峰值检查）
moss_tts.exe --models models --selftest
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--voice <名字>` | 内置音色（`models/voices/*.mossclone`） |
| `--profile <文件>` | 自定义音色档案 |
| `--threads 2` / `--decode-threads 3` | LLM / 解码器的线程数，默认 2+3；想更快试 4+4、4+6。网页「性能」面板也能改（写进 `userdata/settings.json`），改完点面板上的「重启引擎」 |
| `--chunk 8` | 解码块大小（帧）；越小首音越快，4 帧≈21 ms 首块 |
| `--seed 0` | 采样种子（同一 seed 输出确定） |
| `--greedy` | 关掉采样（完全确定，但**不会自然结束**，要配 `--frames`） |
| `--frames 375` | 每段帧上限（80 ms/帧） |
| `--no-cache` | 不用前缀 KV 缓存 |

## 音色克隆

最省事的做法是打开本地页面（双击 `moss_tts.exe` 会自动打开），在「克隆音色」里
**把音频文件拖进去**：MP3 / WAV / M4A / OGG / FLAC 都行（浏览器在本机解码，不上传）。
拖进去之后可以在波形上拖两端选片段、点「试听这一段」听一遍，再创建音色。

命令行等价物：

```bat
moss_clone.exe --encoder models/encoder_f16.moss --in my_voice.wav --out models/voices/me.mossclone --name me
moss_clone.exe --encoder models/encoder_f16.moss --in long.mp3.wav --start 12 --seconds 8 --out models/voices/me.mossclone

moss_tts.exe --models models --profile models/voices/me.mossclone --text "你好，这是我的声音。" --out me.wav
```

参考音频要求：**PCM WAV**（16/24/32-bit 或 float32），任意采样率（内部重采样到 48 kHz），
单声道会自动复制成立体声。**最长取 20 秒**（`--start` 选起点、`--seconds` 选长度，超出部分自动截掉）；
5~15 秒单人、无背景音乐的片段效果最好。
命令行只吃 PCM：MP3/AAC/OGG 这类压缩格式先用工具转成 WAV（网页那条路不用管，浏览器会解码）。

`.mossclone` 只有几 KB（16 路 × N 帧音频码），可以直接发给别人；
对方再用 `--profile` 就能复现同一种音色。

## 接入自己的程序（C API）

`include/moss/tts.h` + `lib/moss_tts.lib`（静态库，同样零依赖）：

```c
#include "moss/tts.h"

void on_pcm(void *ud, const int16_t *pcm, int frames, int sample_rate) {
    /* 写声卡 / 发网络 / 塞环形缓冲 */
}

moss_engine *e = moss_create("models/llm_q8x4.moss", "models/decoder_f16.moss",
                             "models/encoder_f16.moss", "models/tokenizer.mossvocab",
                             "models/voices", "models/cache", 4, 6);
moss_set_voice(e, "Junhao");
moss_speak(e, "你好，世界。", on_pcm, NULL, 0, 0, 375, 8);
moss_destroy(e);
```

- 回调在 `moss_speak` 的调用线程上同步触发，块大小约 `chunk × 80 ms`。
- 首块延迟实测 ~130 ms（`--chunk 8`），`--chunk 4` 可以更低。
- 音色克隆走 `moss_clone(e, "ref.wav", "me.mossclone")`。

## 实测（i5-14600KF，4+6 线程）

| 场景 | 首音 | 总耗时 | RTF |
|---|---|---|---|
| 5.12 s 中文（64 帧） | 128 ms | 738 ms | 0.144 |
| 11.2 s 中文（140 帧） | 149 ms | 1687 ms | 0.151 |
| 26.3 s 长文本（2 段，324 帧） | 168 ms | 4053 ms | 0.154 |

- 音色前缀 KV 缓存后，每段的 prefill 只有 30~60 ms（不缓存要 175 ms）。
- 解码器比 LLM 慢（约 8 ms/帧 vs 6.4 ms/帧），所以 `--decode-threads` 可以比 `--threads` 大。

## 目录

```
moss_tts.exe        文本 -> 音频（WAV / 裸 PCM）
moss_clone.exe      参考音频 -> .mossclone
models/             权重 + 词表 + 内置音色（首次运行会生成 models/cache/*.mosskv）
include/moss/tts.h  C API
lib/moss_tts.lib    静态库
```

## 已知边界

- 上下文上限 768 个 token：超出会**自动分块**（按 75 token 切句），块间插入 0.24~0.40 s 停顿。
- 文本不做语义归一化（数字、单位、日期都原样喂给模型）；需要的话请在调用侧先处理。
- 纯 `--greedy` 模式模型不会自己结束，必须用 `--frames` 限长。
