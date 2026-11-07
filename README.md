# MOSS-TTS CPU

MOSS-TTS-Nano-100M 的原生 **C++ CPU 推理引擎**（Windows x64，零第三方运行时依赖）。
文本进、音频出，流式出声，支持音色克隆；全部在本机计算，不联网、不上传文本。

> 最终形态是 **DeepSeek Harness 的 TTS 插件**（在
> [litterbit/dsh-plugin-moss-tts](https://github.com/litterbit/dsh-plugin-moss-tts)）；
> 这个仓库是它背后的引擎，同时提供可独立使用的命令行、本地 HTTP 服务和一个浏览器试听台
> ——三者跑的是同一个二进制、同一份权重。

## 装进 DeepSeek Harness

```bash
dsh plugin --profile desktop add github:litterbit/dsh-plugin-moss-tts#main
```

重启 DSH，在 **设置 → Moss 语音** 里点「下载引擎」。已经用本页的 zip 装过？

插件会直接复用（同一个 `%LOCALAPPDATA%\MossTTS`），不重复下载。

## 下载

**[最新版本](https://github.com/litterbit/dsh-moss-tts/releases/latest)** · [项目门户](https://litterbit.github.io/dsh-moss-tts/)

| 包 | 大小 | 说明 |
|---|---|---|
| `MossTTS-win-x64-full.zip` | ~159 MB | 程序 + 全部权重 + 18 个内置音色，解压即用 |
| `MossTTS-win-x64-lite.zip` | ~1 MB | 只有程序与网页，权重另取（给插件/开发者） |
| `MossTTS-sdk.zip` | ~1 MB | `tts.h` + `moss_tts.lib` + API 文档 |

三步开始：下载 → 解压 → 双击 `moss_tts.exe`（无参数 = 起本地服务并打开浏览器）。
首次运行 Windows 可能提示"已保护你的电脑"（未购买代码签名证书）：更多信息 → 仍要运行。

## 性能（i5-14600KF，6P+8E，AVX2）

| 项 | 数值 |
|---|---|
| RTF（5.12 s 音频） | **0.149**（4+6 线程，热缓存） |
| 首块出声 | 热缓存 ~100 ms（实测 93–114 ms），冷启动约 230 ms |
| LLM 解码 | 7.4 ms/帧（W8A16），prefill 167–176 ms |
| 音频解码器 | 5.9 ms/帧（fp16，流式 chunk 8） |
| 内存 / 依赖 | ~250 MB / 只有 KERNEL32、WS2_32、SHELL32 |

精度：teacher forcing 下与 f16 参考相比 code 翻转 5.2%、KL 0.00145（听感无差异）。
量化下限是 8 bit：Q4 全量化翻转 64~86%、响度失控，已排除。

## 用法

**本地服务 + 浏览器**

```bat
moss_tts.exe                     :: 起服务，自动打开 http://127.0.0.1:8759/
moss_tts.exe --serve --port 9000 --no-open
moss_tts.exe --stdio             :: 给插件用：stdin/stdout 上的 JSON 行协议，不占端口
```

试听台默认勾着「边播边算（省 CPU）」：推理只比播放位置快 1.5 秒，长文本听着听着就生成完了，
CPU 平时只忙 15% 左右；取消勾选则一次算完（短句、或者急着要文件时更快）。

装好之后，门户页上的「在网页里直接试听」也能用：页面会自己探测 `127.0.0.1`（8759→8763）
并连上本机引擎；连不上就提示去下载。默认只放行本项目这一个 origin，
`--allow-origin none` 可彻底关掉跨源。

**命令行**

```bat
moss_tts.exe --text "你好，世界。" --voice Junhao --out hello.wav
moss_tts.exe --text "..." --pcm-raw - > out.pcm      :: s16le / 48k / 立体声
moss_tts.exe --list-voices
moss_tts.exe --selftest
moss_clone.exe --encoder models/encoder_f16.moss --in ref.wav --out models/voices/me.mossclone --name me
moss_tts.exe --profile models/voices/me.mossclone --text "用自己的音色说话。" --out me.wav
```

**HTTP（只监听 127.0.0.1）**

```
GET  /api/v1/health | /info | /voices
POST /api/v1/tts/stream      JSON -> chunked s16le（边算边拿）
POST /api/v1/tts/wav         JSON -> 完整 WAV
POST /api/v1/voices/clone?name=xxx
GET  /api/v1/jobs/{id}       统计（首块/RTF/帧数/缓存命中）
POST /api/v1/jobs/{id}/cancel
```

完整协议见 [`doc/api.md`](doc/api.md)，C 接口见 [`csrc/include/moss/tts.h`](csrc/include/moss/tts.h)。

## 从源码构建

```bat
cd csrc
build.bat                     :: 需要 VS 2022（Build Tools 即可）+ C++ 工作负载
```

产物在 `csrc/build/`。权重不进仓库，由 `tools/export_*.py` 从 PyTorch checkpoint 导出，
或用 `scripts/fetch_weights.ps1` 从 [HuggingFace](https://huggingface.co/litterbit/moss-tts-nano-moss)
按 `scripts/model_lock.json` 里固定的 revision 拉取并逐个校验（布局与更新流程见 [`doc/hf.md`](doc/hf.md)）。

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/fetch_weights.ps1 -Weights weights
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/pack_release.ps1 -Mode all -Tag v0.1.1
python tools/http_smoke.py                          :: 服务端冒烟（流式/取消/安全/限速）
node tools/playback_sim.mjs                         :: 播放链路重放（不能丢样）
```

## 目录

| 路径 | 内容 |
|---|---|
| `csrc/` | C++ 引擎：LLM / 解码器 / 编码器 / 分词器 / Runtime / HTTP 服务 / C ABI |
| `web/app/` | 本地试听台（随包发布，由服务托管） |
| `web/landing/` | GitHub Pages 门户 |
| `tools/` | 权重导出、ONNX 对拍、冒烟测试、量化扫描（Python，离线用） |
| `scripts/` | 权重下载、模型 lock、打包、发布 |
| `doc/` | 设计文档与实测记录（`api.md`、`产品化与分发方案.md`、`v0.1现状与后续.md`、`1103定档-推理栈现状.md`） |

DSH 插件是独立仓库 [litterbit/dsh-plugin-moss-tts](https://github.com/litterbit/dsh-plugin-moss-tts)；
本仓库对它只有一个承诺：`moss_tts.exe --stdio` 的 JSON 行协议（字段只增不改，见 `doc/api.md`）。

## 许可

代码与派生的模型权重均为 **Apache-2.0**：见 [`LICENSE`](LICENSE)、[`NOTICE`](NOTICE)。
上游模型是 MOSS-TTS-Nano-100M；本项目对权重做了 int8 4×4 量化重排与 fp16 编解码器转换，
改动细节写在同一份 NOTICE 里。与模型作者无隶属关系；请只使用你有权使用的声音做克隆。
