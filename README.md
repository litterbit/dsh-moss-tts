# dsh-moss-tts

MOSS-TTS-Nano 的 **CPU 推理引擎**（纯 C++，零第三方依赖），最终形态是 DeepSeek TTS 插件；
推理部分可以完全独立使用：文本进、音频出（整段 WAV 或流式 PCM），也支持参考音频做**音色克隆**。

## 两个入口

**1. 独立发版（推荐先看这个）**

```bat
cd dist\moss-tts-cpu-win64
moss_tts.exe --models models --list-voices
moss_tts.exe --models models --text "你好，我是语音助手。" --voice Junhao --out hello.wav
moss_tts.exe --models models --text "..." --voice Xiaoyu --pcm-raw - > out.pcm
moss_tts.exe --models models --selftest
```

`moss_clone.exe` 把一段 PCM WAV 编码成几 KB 的音色档案，`moss_tts --profile` 用它合成；
C API 见 `include/moss/tts.h`（静态库 `lib/moss_tts.lib`）。打包：`scripts\pack_release.bat zip`。

实测（4+6 线程）：5.12 s 中文 **764 ms、RTF 0.149、首音 128 ms**；ONNX 基线同句 2260 ms。

**2. 开发（本项目里跑）**

```bash
cd csrc && build.bat                      # 产物在 csrc/build/
build\moss_tts.exe --models ..\models --text "你好，世界。" --voice Junhao --out a.wav
build\moss_spmtest.exe --vocab ..\weights\tokenizer.mossvocab --corpus tests\data\spm_corpus.txt --golden tests\data\spm_golden.txt
```

权重导出与对拍工具在 `tools/`（LLM/解码器/编码器/词表/内置音色/ONNX 参考）。ONNX 基线：`scripts/run_baseline.py`。

## 目录

| 路径 | 内容 |
|---|---|
| `csrc/` | C++ 引擎（LLM / 解码器 / 编码器 / 分词器 / 引擎 / C API），`build.bat` 一键构建 |
| `dist/` | 组装好的发布包（170 MB 目录 / 159 MB zip） |
| `tools/` | 权重导出、prompt、ONNX 对拍、SPM 语料、打包辅助 |
| `src/` `scripts/` | Python 侧（ONNX 基线封装、入口脚本） |
| `vendor/` | 上游 ONNX 运行时（原样复制，只读） |
| `doc/` | 设计 + 实测：`推理加速计划.md`（阶段 1~3）、`并行与发版计划.md`（阶段 A~E）、`release_readme.md` |
| `AGENTS.md` | 约定、环境、已知坑、各项实测数字 |

## 关键数字（i5-14600KF）

| 项 | 数值 |
|---|---|
| LLM（Q8_0_4x4） | 6.4 ms/帧，prefill 176 ms（带前缀缓存 34 ms） |
| 解码器（fp16，流式 chunk 8） | 6.2 ms/帧，首块 35 ms，流式与整段**逐位一致** |
| 端到端 RTF | 0.149（5.12 s）/ 0.178（13 s） |
| 音色克隆 | 编码 10 s 参考约 1.2 s，档案几 KB |
| 依赖 | 仅 `KERNEL32.dll`（静态 CRT） |
